use anyhow::{Context, Result};
use base64::Engine;
use clap::Parser;
use futures_util::stream::StreamExt;
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use std::env;
use std::io::{self, Write};
use tokio::fs;

#[derive(Parser)]
#[command(name = "openai-client")]
#[command(about = "OpenAI API Chat Client")]
struct Args {
    /// 指定要读取的文件路径
    #[arg(short = 'f', long = "file")]
    file_path: Option<String>,

    /// 指定要使用的模型名称
    #[arg(short = 'm', long = "model", default_value = "plus")]
    model_name: String,

    /// 显示调试日志
    #[arg(short = 'd', long = "debug")]
    debug_mode: bool,

    /// 指定图片路径，可以多次传递，允许与文本混合
    #[arg(short = 'i', long = "image")]
    image_paths: Vec<String>,

    /// 连接的服务端地址
    #[arg(short = 's', long = "server", default_value = "127.0.0.1:8000")]
    server_host: String,

    /// 用户输入的问题
    #[arg(trailing_var_arg = true)]
    user_input: Vec<String>,
}

#[derive(Clone)]
struct ChatConfig {
    url: String,
    api_key: String,
    model: String,
    temperature: f64,
}

#[derive(Serialize, Debug)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f64,
    stream: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Message {
    role: String,
    content: Vec<ContentItem>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type")]
enum ContentItem {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ImageUrl {
    url: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Option<Vec<Choice>>,
}

#[derive(Deserialize)]
struct Choice {
    delta: Option<Delta>,
}

#[derive(Deserialize)]
struct Delta {
    content: Option<String>,
    reasoning_content: Option<String>,
}

fn get_env_with_default(key: &str, default_value: &str) -> String {
    env::var(key).unwrap_or_else(|_| default_value.to_string())
}

fn load_config(model: &str, server_host: &str) -> Result<ChatConfig> {
    let temp_str = get_env_with_default("OPENAI_TEMPERATURE", "0.7");
    let temperature = temp_str.parse::<f64>()
        .with_context(|| format!("无法解析温度参数: {}", temp_str))?;

    let server_url = format!("http://{}/v1/chat/completions", server_host);
    Ok(ChatConfig {
        url: get_env_with_default("OPENAI_URL", server_url.as_str()),
        api_key: get_env_with_default("OPENAI_API_KEY", "none"),
        model: get_env_with_default("OPENAI_MODEL", model),
        temperature,
    })
}

async fn chat(messages: &[Message], config: &ChatConfig) -> Result<String> {
    let client = reqwest::Client::new();
    
    let request_body = ChatRequest {
        model: config.model.clone(),
        messages: messages.to_vec(),
        temperature: config.temperature,
        stream: true,
    };

    let response = client
        .post(&config.url)
        .header(CONTENT_TYPE, "application/json")
        .header("Accept-Encoding", "identity")
        .header(ACCEPT, "text/event-stream")
        .header(AUTHORIZATION, format!("Bearer {}", config.api_key))
        .json(&request_body)
        .send()
        .await
        .context("HTTP请求失败")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("请求失败，状态码: {}, 响应: {}", status, body);
    }

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut answer = String::new();
    let mut is_in_thinking = false;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("读取SSE流失败")?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(newline_pos) = buffer.find('\n') {
            let line = buffer[..newline_pos].trim().to_string();
            buffer.drain(..=newline_pos);

            if line.starts_with("data:") {
                let data = line[5..].trim();
                if data == "[DONE]" {
                    if is_in_thinking {
                        println!("\n------------------ THINKING END ------------------");
                    }
                    println!();
                    return Ok(answer);
                }

                if let Ok(event) = serde_json::from_str::<ChatResponse>(data) {
                    if let Some(choices) = event.choices {
                        if let Some(choice) = choices.first() {
                            if let Some(delta) = &choice.delta {
                                if let Some(reasoning_content) = &delta.reasoning_content {
                                    if !is_in_thinking {
                                        print!("----------------- THINKING START -----------------\n");
                                        print!("{}", reasoning_content.trim_start());
                                        is_in_thinking = true;
                                    } else {
                                        print!("{}", reasoning_content);
                                    }
                                    io::stdout().flush().unwrap();
                                }
                                if let Some(content) = &delta.content {
                                    if is_in_thinking {
                                        print!("\n------------------ THINKING END ------------------\n\n");
                                        print!("{}", content.trim_start());
                                        is_in_thinking = false;
                                    } else {
                                        print!("{}", content);
                                    }
                                    io::stdout().flush().unwrap();
                                    answer.push_str(content);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    println!();
    Ok(answer)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    println!("* Using model: {}", args.model_name);
    let config = load_config(&args.model_name, &args.server_host)?;
    
    let mut messages: Vec<Message> = Vec::new();

    // 构造 user message 的 content 数组，顺序与参数一致
    let mut content_items: Vec<ContentItem> = Vec::new();

    // 处理文件输入
    if let Some(file_path) = &args.file_path {
        let file_content = fs::read_to_string(file_path)
            .await
            .with_context(|| format!("无法读取文件: {}", file_path))?;

        if args.debug_mode {
            println!("> {}", file_content);
        }
        content_items.push(ContentItem::Text { text: file_content.clone() });
    }

    // 处理命令行参数中的用户输入
    if !args.user_input.is_empty() {
        let user_input = args.user_input.join(" ");
        if args.debug_mode {
            println!("> {}", user_input);
        }
        content_items.push(ContentItem::Text { text: user_input.clone() });
    }

    // 处理图片输入
    for img in &args.image_paths {
        let image_data = fs::read(img)
            .await
            .with_context(|| format!("无法读取图片文件: {}", img))?;

        // 检测文件扩展名以确定MIME类型
        let path = std::path::Path::new(img);
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("jpeg")
            .to_lowercase();

        let mime_type = match ext.as_str() {
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            "gif" => "image/gif",
            "bmp" => "image/bmp",
            "webp" => "image/webp",
            _ => "image/jpeg", // 默认使用 jpeg
        };

        let encoded = base64::prelude::BASE64_STANDARD.encode(&image_data);
        content_items.push(ContentItem::ImageUrl {
            image_url: ImageUrl {
                url: format!("data:{};base64,{}", mime_type, encoded),
            },
        });
    }

    if !content_items.is_empty() {
        messages.push(Message {
            role: "user".to_string(),
            content: content_items,
        });
    }

    if !messages.is_empty() {
        let first_answer = chat(&messages, &config).await?;
        messages.push(Message {
            role: "assistant".to_string(),
            content: vec![ContentItem::Text { text: first_answer }],
        });
    }

    // 交互式对话循环
    println!("开始对话（输入'exit'退出）：");

    let mut rl = rustyline::DefaultEditor::new().context("无法初始化输入编辑器")?;

    loop {
        let res = rl.readline("> ");
        match res {
            Ok(line) => {
                let input = line.trim();
                if input == "exit" {
                    break;
                }
                if input.is_empty() {
                    continue;
                }

                // 添加到历史记录
                let _ = rl.add_history_entry(&line);

                messages.push(Message {
                    role: "user".to_string(),
                    content: vec![ContentItem::Text { text: input.to_string() }],
                });

                let answer = chat(&messages, &config).await?;
                messages.push(Message {
                    role: "assistant".to_string(),
                    content: vec![ContentItem::Text { text: answer }],
                });
            }
            Err(rustyline::error::ReadlineError::Interrupted) | Err(rustyline::error::ReadlineError::Eof) => {
                break;
            }
            Err(e) => {
                eprintln!("读取输入失败: {}", e);
                break;
            }
        }
    }

    Ok(())
}