use anyhow::{Context, Result};
use base64::Engine;
use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers, KeyboardEnhancementFlags, PushKeyboardEnhancementFlags};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use crossterm::execute;
use futures_util::stream::StreamExt;
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use std::env;
use std::io::{self, Write};
use std::time::{Duration, Instant};
use tokio::fs;
use unicode_width::UnicodeWidthChar;

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

/// Redraw the current input line in place, positioning the terminal cursor at
/// `cursor_char_pos` (a char-index into `line`).
/// `prompt` must be ASCII so its byte-len == display-width.
fn redraw_current_line(prompt: &str, line: &str, cursor_char_pos: usize) -> io::Result<()> {
    // Go to start of the terminal line
    print!("\r");
    // Prompt
    print!("{}", prompt);
    // Full line content
    print!("{}", line);
    // Erase to end of line (clears any leftover characters after a deletion)
    print!("\x1b[K");
    // Move the cursor back to the right position
    let after_width: usize = line.chars()
        .skip(cursor_char_pos)
        .map(|c| c.width().unwrap_or(1))
        .sum();
    if after_width > 0 {
        print!("\x1b[{}D", after_width);
    }
    io::stdout().flush()
}

fn read_multiline_input() -> Result<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut stdout = io::stdout();
    let mut current_line = String::new();
    // Cursor position as a char-index within `current_line`
    let mut cursor_char_pos: usize = 0;
    // Paste detection: record the time of the last key event.
    // Initialised far in the past so the very first Enter always counts as a real send.
    let mut last_key_time: Instant = Instant::now() - Duration::from_secs(10);
    // An Enter pressed less than this after the previous key is treated as part of a paste.
    const PASTE_IDLE_THRESHOLD: Duration = Duration::from_millis(100);
    
    enable_raw_mode().context("无法启用 raw mode")?;
    
    // 尝试启用键盘增强模式以支持 Shift+Enter
    // 只启用 DISAMBIGUATE_ESCAPE_CODES，避免干扰正常字符输入
    let _ = execute!(
        stdout,
        PushKeyboardEnhancementFlags(
            KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
        )
    );
    
    print!("> ");
    stdout.flush()?;
    
    loop {
        if let Event::Key(KeyEvent { code, modifiers, .. }) = event::read()? {
            // Timestamp this key event for paste detection
            let now = Instant::now();
            let time_since_last = now.duration_since(last_key_time);
            last_key_time = now;

            // Determine the prompt for the current terminal line
            let prompt = if lines.is_empty() { "> " } else { "  " };

            match code {
                KeyCode::Enter if modifiers.contains(KeyModifiers::SHIFT) => {
                    // Shift+Enter: 换行
                    lines.push(current_line.clone());
                    current_line.clear();
                    cursor_char_pos = 0;
                    print!("\r\n  ");
                    stdout.flush()?;
                }
                KeyCode::Enter if modifiers.contains(KeyModifiers::ALT) => {
                    // Alt+Enter (Option+Enter on Mac): 换行 - 更可靠的替代方案
                    lines.push(current_line.clone());
                    current_line.clear();
                    cursor_char_pos = 0;
                    print!("\r\n  ");
                    stdout.flush()?;
                }
                KeyCode::Enter => {
                    // If the previous key arrived very recently (< 100 ms), this Enter
                    // is almost certainly part of a paste — treat it as a newline.
                    // Only a "slow" Enter (idle >= 100 ms) actually sends the message.
                    if time_since_last < PASTE_IDLE_THRESHOLD {
                        // Paste newline: same as Shift+Enter
                        lines.push(current_line.clone());
                        current_line.clear();
                        cursor_char_pos = 0;
                        print!("\r\n  ");
                        stdout.flush()?;
                    } else {
                        // Real Enter: send message
                        lines.push(current_line.clone());
                        print!("\r\n");  // 使用 \r\n 确保光标回到行首
                        stdout.flush()?;
                        break;
                    }
                }
                KeyCode::Char(c) if modifiers.contains(KeyModifiers::CONTROL) && c == 'c' => {
                    // Ctrl+C: 退出
                    let _ = execute!(stdout, crossterm::event::PopKeyboardEnhancementFlags);
                    disable_raw_mode()?;
                    return Err(anyhow::anyhow!("输入被取消"));
                }
                KeyCode::Char('u') if modifiers.contains(KeyModifiers::CONTROL) => {
                    // Ctrl+U: 删除光标前的所有内容
                    let chars: Vec<char> = current_line.chars().collect();
                    current_line = chars[cursor_char_pos..].iter().collect();
                    cursor_char_pos = 0;
                    redraw_current_line(prompt, &current_line, cursor_char_pos)
                        .context("重绘输入行失败")?;
                }
                KeyCode::Char('k') if modifiers.contains(KeyModifiers::CONTROL) => {
                    // Ctrl+K: 删除光标后的所有内容
                    let chars: Vec<char> = current_line.chars().collect();
                    current_line = chars[..cursor_char_pos].iter().collect();
                    redraw_current_line(prompt, &current_line, cursor_char_pos)
                        .context("重绘输入行失败")?;
                }
                KeyCode::Char('a') if modifiers.contains(KeyModifiers::CONTROL) => {
                    // Ctrl+A: 移动光标到行首
                    cursor_char_pos = 0;
                    redraw_current_line(prompt, &current_line, cursor_char_pos)
                        .context("重绘输入行失败")?;
                }
                KeyCode::Char('e') if modifiers.contains(KeyModifiers::CONTROL) => {
                    // Ctrl+E: 移动光标到行尾
                    cursor_char_pos = current_line.chars().count();
                    redraw_current_line(prompt, &current_line, cursor_char_pos)
                        .context("重绘输入行失败")?;
                }
                KeyCode::Char(c) => {
                    // Insert character at cursor position
                    let chars: Vec<char> = current_line.chars().collect();
                    current_line = chars[..cursor_char_pos]
                        .iter()
                        .chain(std::iter::once(&c))
                        .chain(chars[cursor_char_pos..].iter())
                        .collect();
                    cursor_char_pos += 1;
                    redraw_current_line(prompt, &current_line, cursor_char_pos)
                        .context("重绘输入行失败")?;
                }
                KeyCode::Backspace => {
                    if cursor_char_pos > 0 {
                        // Remove the char just before the cursor
                        let chars: Vec<char> = current_line.chars().collect();
                        cursor_char_pos -= 1;
                        current_line = chars[..cursor_char_pos]
                            .iter()
                            .chain(chars[cursor_char_pos + 1..].iter())
                            .collect();
                        redraw_current_line(prompt, &current_line, cursor_char_pos)
                            .context("重绘输入行失败")?;
                    } else if !lines.is_empty() {
                        // 合并到上一行
                        let prev_line = lines.pop().unwrap();
                        let prev_prompt = if lines.is_empty() { "> " } else { "  " };
                        let merged = prev_line + &current_line;
                        cursor_char_pos = merged.chars().count() - current_line.chars().count();
                        current_line = merged;
                        // 向上移动一个终端行后重绘
                        print!("\x1b[A");
                        redraw_current_line(prev_prompt, &current_line, cursor_char_pos)
                            .context("重绘输入行失败")?;
                    }
                }
                KeyCode::Left => {
                    if cursor_char_pos > 0 {
                        cursor_char_pos -= 1;
                        // Move terminal cursor left by the width of the character we just passed
                        let chars: Vec<char> = current_line.chars().collect();
                        let w = chars[cursor_char_pos].width().unwrap_or(1);
                        print!("\x1b[{}D", w);
                        stdout.flush()?;
                    }
                }
                KeyCode::Right => {
                    let chars: Vec<char> = current_line.chars().collect();
                    if cursor_char_pos < chars.len() {
                        let w = chars[cursor_char_pos].width().unwrap_or(1);
                        cursor_char_pos += 1;
                        // Move terminal cursor right by the width of the character we just passed
                        print!("\x1b[{}C", w);
                        stdout.flush()?;
                    }
                }
                KeyCode::Esc => {
                    let _ = execute!(stdout, crossterm::event::PopKeyboardEnhancementFlags);
                    disable_raw_mode()?;
                    return Err(anyhow::anyhow!("输入被取消"));
                }
                _ => {}
            }
        }
    }
    
    let _ = execute!(stdout, crossterm::event::PopKeyboardEnhancementFlags);
    disable_raw_mode().context("无法禁用 raw mode")?;
    
    // 将所有行用换行符连接（每行内容已经是纯粹的用户输入，不包含提示符）
    Ok(lines.join("\n"))
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
                                    if !reasoning_content.is_empty() {
                                        if !is_in_thinking {
                                            print!("----------------- THINKING START -----------------\n");
                                            print!("{}", reasoning_content.trim_start());
                                            is_in_thinking = true;
                                        } else {
                                            print!("{}", reasoning_content);
                                        }
                                        io::stdout().flush().unwrap();
                                    }
                                }
                                if let Some(content) = &delta.content {
                                    if !content.is_empty() {
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
    println!("开始对话（输入'exit'退出）");
    println!("提示：按 Option+Enter (Alt+Enter) 换行，Enter 发送消息");
    println!("      在支持键盘增强的终端（iTerm2, WezTerm）中也可使用 Shift+Enter");
    println!();

    loop {
        let input = match read_multiline_input() {
            Ok(text) => text.trim().to_string(),
            Err(_) => break,
        };

        if input == "exit" {
            break;
        }
        if input.is_empty() {
            continue;
        }

        messages.push(Message {
            role: "user".to_string(),
            content: vec![ContentItem::Text { text: input }],
        });

        let answer = chat(&messages, &config).await?;
        messages.push(Message {
            role: "assistant".to_string(),
            content: vec![ContentItem::Text { text: answer }],
        });
    }

    Ok(())
}