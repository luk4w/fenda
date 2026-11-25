//! Cliente HTTP do comando `ask` para servidores compatíveis com a API de chat da OpenAI (vLLM, llama.cpp, LM Studio, Ollama). Sem TLS; destinado a servidores locais.

use serde_json::{Value, json};
use std::error::Error;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

type Res<T> = Result<T, Box<dyn Error>>;

pub struct ChatClient {
    host: String,
    port: u16,
}

impl ChatClient {
    pub fn new(host: &str, port: u16) -> Self {
        ChatClient {
            host: host.to_string(),
            port,
        }
    }

    pub fn chat(&self, model: &str, system: &str, user: &str) -> Res<(String, Option<u64>)> {
        let mut messages = Vec::new();
        if !system.is_empty() {
            messages.push(json!({"role": "system", "content": system}));
        }
        messages.push(json!({"role": "user", "content": user}));
        let body = json!({
            "model": model,
            "stream": false,
            "temperature": 0.0,
            "messages": messages,
        })
        .to_string();
        let (status, text) = self.post("/v1/chat/completions", &body)?;
        if status != 200 {
            let shown: String = text.chars().take(300).collect();
            return Err(format!("chat server returned HTTP {status}: {shown}").into());
        }
        if text.trim_start().starts_with("data:") {
            let mut content = String::new();
            let mut read_tokens = None;
            for line in text.lines() {
                let Some(data) = line.trim().strip_prefix("data:") else {
                    continue;
                };
                let Ok(piece) = serde_json::from_str::<Value>(data.trim()) else {
                    continue;
                };
                if let Some(delta) = piece["choices"][0]["delta"]["content"].as_str() {
                    content.push_str(delta);
                }
                read_tokens = piece["usage"]["prompt_tokens"].as_u64().or(read_tokens);
            }
            if content.is_empty() {
                return Err("chat server: the stream carried no content".into());
            }
            return Ok((content, read_tokens));
        }
        let reply: Value = serde_json::from_str(&text)?;
        let content = reply["choices"][0]["message"]["content"]
            .as_str()
            .ok_or("chat server: no message content in the reply")?;
        Ok((
            content.to_string(),
            reply["usage"]["prompt_tokens"].as_u64(),
        ))
    }

    fn post(&self, path: &str, body: &str) -> Res<(u16, String)> {
        let mut stream = TcpStream::connect((self.host.as_str(), self.port)).map_err(|e| {
            format!(
                "could not reach the model server at {}:{}: {e}",
                self.host, self.port
            )
        })?;
        stream.set_read_timeout(Some(Duration::from_secs(600)))?;
        write!(
            stream,
            "POST {path} HTTP/1.1\r\nHost: {}:{}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            self.host,
            self.port,
            body.len()
        )?;
        stream.write_all(body.as_bytes())?;

        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let status: u16 = line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .ok_or("model server: malformed HTTP reply")?;

        let mut length: Option<usize> = None;
        let mut chunked = false;
        loop {
            line.clear();
            if reader.read_line(&mut line)? == 0 || line.trim().is_empty() {
                break;
            }
            if let Some((name, value)) = line.split_once(':') {
                let value = value.trim();
                if name.eq_ignore_ascii_case("content-length") {
                    length = value.parse().ok();
                } else if name.eq_ignore_ascii_case("transfer-encoding") {
                    chunked = value.to_ascii_lowercase().contains("chunked");
                }
            }
        }

        let mut raw = Vec::new();
        if chunked {
            loop {
                line.clear();
                if reader.read_line(&mut line)? == 0 {
                    break;
                }
                let size = usize::from_str_radix(line.trim().split(';').next().unwrap_or(""), 16)
                    .map_err(|_| "model server: malformed chunk")?;
                if size == 0 {
                    break;
                }
                let start = raw.len();
                raw.resize(start + size, 0);
                reader.read_exact(&mut raw[start..])?;
                line.clear();
                reader.read_line(&mut line)?;
            }
        } else if let Some(n) = length {
            raw.resize(n, 0);
            reader.read_exact(&mut raw)?;
        } else {
            reader.read_to_end(&mut raw)?;
        }
        Ok((status, String::from_utf8_lossy(&raw).into_owned()))
    }
}
