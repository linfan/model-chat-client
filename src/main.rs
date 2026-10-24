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
use uuid::Uuid;

#[derive(Parser)]
#[command(name = "chat-client")]
#[command(about = "Model Chat Client")]
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

fn char_width(c: char) -> usize {
    c.width().unwrap_or(1)
}

/// Advance through one terminal line, wrapping when a character does not fit.
/// `col == width` means the terminal is in its "wrap pending" state (the next
/// printable character moves to the following row).
fn advance_cursor(row: &mut usize, col: &mut usize, width: usize, w: usize) {
    if w == 0 {
        return;
    }
    if *col + w > width {
        *row += 1;
        *col = 0;
    }
    *col += w;
}

fn rendered_rows(prompt: &str, line: &str, width: usize) -> usize {
    let mut row = 0;
    let mut col = 0;
    for c in prompt.chars().chain(line.chars()) {
        advance_cursor(&mut row, &mut col, width, char_width(c));
    }
    row + 1
}

struct LineRender {
    /// Row of the cursor relative to the first row of this line.
    cursor_row: usize,
    /// Number of terminal rows occupied by prompt + line.
    total_rows: usize,
}

/// Redraw the current input line in place, positioning the terminal cursor at
/// `cursor_char_pos` (a char-index into `line`).
/// `prev_cursor_row` is the `cursor_row` returned by the previous render; it is
/// used to move from the cursor back to the first row of the line.
fn redraw_current_line(
    prompt: &str,
    line: &str,
    cursor_char_pos: usize,
    prev_cursor_row: usize,
) -> io::Result<LineRender> {
    let width = crossterm::terminal::size()
        .map(|(w, _)| w as usize)
        .unwrap_or(80)
        .max(1);

    // Walk the rendered text up to the cursor to find its terminal position.
    let mut row = 0;
    let mut col = 0;
    for c in prompt.chars() {
        advance_cursor(&mut row, &mut col, width, char_width(c));
    }
    for c in line.chars().take(cursor_char_pos) {
        advance_cursor(&mut row, &mut col, width, char_width(c));
    }
    // If the cursor lands right after a full row and text follows, the next
    // character starts on the following row.
    let has_text_after = line.chars().skip(cursor_char_pos).any(|c| char_width(c) > 0);
    let (target_row, target_col) = if has_text_after && col >= width {
        (row + 1, 0)
    } else {
        (row, col.min(width - 1))
    };
    let total_rows = rendered_rows(prompt, line, width);

    let mut out = io::stdout();
    // Back to the first row of the input line, then wipe it and everything below
    // it so a wrapped line is never appended to its previous rendering.
    if prev_cursor_row > 0 {
        write!(out, "\x1b[{}A", prev_cursor_row)?;
    }
    write!(out, "\r\x1b[J{}{}", prompt, line)?;
    // The cursor is now at the last rendered row; walk up to the first one.
    if total_rows > 1 {
        write!(out, "\x1b[{}A", total_rows - 1)?;
    }
    write!(out, "\r")?;
    if target_row > 0 {
        write!(out, "\x1b[{}B", target_row)?;
    }
    if target_col > 0 {
        write!(out, "\x1b[{}C", target_col)?;
    }
    out.flush()?;

    Ok(LineRender { cursor_row: target_row, total_rows })
}

/// Commit `current_line` and start a fresh one below it.
fn start_new_input_line(
    lines: &mut Vec<String>,
    line_rows: &mut Vec<usize>,
    current_line: &mut String,
    cursor_char_pos: &mut usize,
    cursor_row: &mut usize,
    prompt: &str,
) -> io::Result<()> {
    let render = redraw_current_line(
        prompt,
        current_line,
        current_line.chars().count(),
        *cursor_row,
    )?;
    print!("\r\n  ");
    io::stdout().flush()?;
    line_rows.push(render.total_rows);
    lines.push(std::mem::take(current_line));
    *cursor_char_pos = 0;
    *cursor_row = 0;
    Ok(())
}

fn read_multiline_input() -> Result<String> {
    let mut lines: Vec<String> = Vec::new();
    // Number of terminal rows occupied by each completed line, used to move the
    // cursor back up when a line is merged into the previous one.
    let mut line_rows: Vec<usize> = Vec::new();
    let mut stdout = io::stdout();
    let mut current_line = String::new();
    // Cursor position as a char-index within `current_line`
    let mut cursor_char_pos: usize = 0;
    // Cursor row within the current line's (possibly wrapped) rendering
    let mut cursor_row: usize = 0;
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
                    start_new_input_line(
                        &mut lines,
                        &mut line_rows,
                        &mut current_line,
                        &mut cursor_char_pos,
                        &mut cursor_row,
                        prompt,
                    )?;
                }
                KeyCode::Enter if modifiers.contains(KeyModifiers::ALT) => {
                    // Alt+Enter (Option+Enter on Mac): 换行 - 更可靠的替代方案
                    start_new_input_line(
                        &mut lines,
                        &mut line_rows,
                        &mut current_line,
                        &mut cursor_char_pos,
                        &mut cursor_row,
                        prompt,
                    )?;
                }
                KeyCode::Enter => {
                    // If the previous key arrived very recently (< 100 ms), this Enter
                    // is almost certainly part of a paste — treat it as a newline.
                    // Only a "slow" Enter (idle >= 100 ms) actually sends the message.
                    if time_since_last < PASTE_IDLE_THRESHOLD {
                        // Paste newline: same as Shift+Enter
                        start_new_input_line(
                            &mut lines,
                            &mut line_rows,
                            &mut current_line,
                            &mut cursor_char_pos,
                            &mut cursor_row,
                            prompt,
                        )?;
                    } else {
                        // Real Enter: send message
                        redraw_current_line(
                            prompt,
                            &current_line,
                            current_line.chars().count(),
                            cursor_row,
                        )?;
                        print!("\r\n");  // 使用 \r\n 确保光标回到行首
                        stdout.flush()?;
                        lines.push(std::mem::take(&mut current_line));
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
                    let render =
                        redraw_current_line(prompt, &current_line, cursor_char_pos, cursor_row)
                            .context("重绘输入行失败")?;
                    cursor_row = render.cursor_row;
                }
                KeyCode::Char('k') if modifiers.contains(KeyModifiers::CONTROL) => {
                    // Ctrl+K: 删除光标后的所有内容
                    let chars: Vec<char> = current_line.chars().collect();
                    current_line = chars[..cursor_char_pos].iter().collect();
                    let render =
                        redraw_current_line(prompt, &current_line, cursor_char_pos, cursor_row)
                            .context("重绘输入行失败")?;
                    cursor_row = render.cursor_row;
                }
                KeyCode::Char('a') if modifiers.contains(KeyModifiers::CONTROL) => {
                    // Ctrl+A: 移动光标到行首
                    cursor_char_pos = 0;
                    let render =
                        redraw_current_line(prompt, &current_line, cursor_char_pos, cursor_row)
                            .context("重绘输入行失败")?;
                    cursor_row = render.cursor_row;
                }
                KeyCode::Char('e') if modifiers.contains(KeyModifiers::CONTROL) => {
                    // Ctrl+E: 移动光标到行尾
                    cursor_char_pos = current_line.chars().count();
                    let render =
                        redraw_current_line(prompt, &current_line, cursor_char_pos, cursor_row)
                            .context("重绘输入行失败")?;
                    cursor_row = render.cursor_row;
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
                    let render =
                        redraw_current_line(prompt, &current_line, cursor_char_pos, cursor_row)
                            .context("重绘输入行失败")?;
                    cursor_row = render.cursor_row;
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
                        let render =
                            redraw_current_line(prompt, &current_line, cursor_char_pos, cursor_row)
                                .context("重绘输入行失败")?;
                        cursor_row = render.cursor_row;
                    } else if let Some(prev_line) = lines.pop() {
                        // 合并到上一行
                        let prev_rows = line_rows.pop().unwrap_or(1);
                        let prev_prompt = if lines.is_empty() { "> " } else { "  " };
                        let prev_len = prev_line.chars().count();
                        let merged = prev_line + &current_line;
                        cursor_char_pos = prev_len;
                        current_line = merged;
                        // Move the cursor to the first row of the previous line,
                        // which sits `prev_rows` above this line's first row.
                        let up = prev_rows + cursor_row;
                        if up > 0 {
                            print!("\x1b[{}A", up);
                            stdout.flush()?;
                        }
                        let render =
                            redraw_current_line(prev_prompt, &current_line, cursor_char_pos, 0)
                                .context("重绘输入行失败")?;
                        cursor_row = render.cursor_row;
                    }
                }
                KeyCode::Left => {
                    if cursor_char_pos > 0 {
                        cursor_char_pos -= 1;
                        let render =
                            redraw_current_line(prompt, &current_line, cursor_char_pos, cursor_row)
                                .context("重绘输入行失败")?;
                        cursor_row = render.cursor_row;
                    }
                }
                KeyCode::Right => {
                    if cursor_char_pos < current_line.chars().count() {
                        cursor_char_pos += 1;
                        let render =
                            redraw_current_line(prompt, &current_line, cursor_char_pos, cursor_row)
                                .context("重绘输入行失败")?;
                        cursor_row = render.cursor_row;
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

async fn chat(messages: &[Message], config: &ChatConfig, session_id: &str) -> Result<String> {
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
        .header("X-Session-Id", session_id)
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
    let session_id = Uuid::new_v4().to_string();

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
        let first_answer = chat(&messages, &config, &session_id).await?;
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

        let answer = chat(&messages, &config, &session_id).await?;
        messages.push(Message {
            role: "assistant".to_string(),
            content: vec![ContentItem::Text { text: answer }],
        });
    }

    Ok(())
}