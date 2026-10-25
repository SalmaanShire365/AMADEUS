//! Ollama HTTP client: streaming `/api/chat` plus the small read-only
//! endpoints the health check and doctor use.
//!
//! Cancel has to work while Ollama is still loading a model, when no bytes
//! arrive for tens of seconds. So the HTTP read happens on its own thread and
//! the caller waits on a channel with a short timeout, checking the cancel
//! flag between waits. On cancel the caller returns at once; the reader
//! thread notices the flag at its next line (or at the stall timeout), drops
//! the connection, and Ollama stops generating.

use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

use serde_json::{json, Value};

/// How long the stream may go without a single byte before it's abandoned.
/// Long enough for a cold load of a 14B model from disk.
pub fn stall_timeout() -> Duration {
    let secs = std::env::var("AMADEUS_OLLAMA_TIMEOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);
    Duration::from_secs(secs)
}

#[derive(Debug, PartialEq)]
pub enum ChatError {
    Cancelled,
    Unreachable(String),
    Status(u16, String),
    /// Ollama answered with `{"error": …}`.
    Model(String),
    Stream(String),
}

impl std::fmt::Display for ChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatError::Cancelled => write!(f, "cancelled"),
            ChatError::Unreachable(e) => {
                write!(f, "cannot reach Ollama ({e}). Is `ollama serve` running?")
            }
            ChatError::Status(code, d) => write!(f, "ollama returned {code}: {d}"),
            ChatError::Model(e) => write!(f, "ollama: {e}"),
            ChatError::Stream(e) => write!(f, "stream broke: {e}"),
        }
    }
}

pub enum Delta<'a> {
    Content(&'a str),
    /// Reasoning tokens, when a model thinks despite `think: false`. Only
    /// the fact that it's thinking is surfaced; the text is in `Reply`.
    Thinking,
}

#[derive(Debug, Default)]
pub struct Reply {
    pub content: String,
    pub thinking: String,
    pub tool_calls: Vec<Value>,
}

enum Msg {
    Line(String),
    Fail(ChatError),
    Eof,
}

fn agent(read_timeout: Duration) -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(5))
        .timeout_read(read_timeout)
        .build()
}

/// Stream one chat turn. `body` is the full `/api/chat` request; `stream` is
/// forced on.
pub fn chat(
    base_url: &str,
    mut body: Value,
    cancel: &Arc<AtomicBool>,
    mut on_delta: impl FnMut(Delta),
) -> Result<Reply, ChatError> {
    body["stream"] = json!(true);
    let url = format!("{}/api/chat", base_url.trim_end_matches('/'));
    let (tx, rx) = mpsc::channel::<Msg>();
    let flag = cancel.clone();
    let stall = stall_timeout();

    thread::spawn(move || {
        let resp = match agent(stall).post(&url).send_json(body) {
            Ok(r) => r,
            Err(ureq::Error::Status(code, r)) => {
                let detail = r.into_string().unwrap_or_default();
                let detail = serde_json::from_str::<Value>(&detail)
                    .ok()
                    .and_then(|v| v["error"].as_str().map(str::to_string))
                    .unwrap_or(detail);
                let _ = tx.send(Msg::Fail(ChatError::Status(code, detail.trim().into())));
                return;
            }
            Err(e) => {
                let _ = tx.send(Msg::Fail(ChatError::Unreachable(e.to_string())));
                return;
            }
        };
        let mut reader = BufReader::new(resp.into_reader());
        let mut line = String::new();
        loop {
            if flag.load(Ordering::Relaxed) {
                return; // dropping the reader closes the connection
            }
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => {
                    let _ = tx.send(Msg::Eof);
                    return;
                }
                Ok(_) => {
                    if tx.send(Msg::Line(line.clone())).is_err() {
                        return;
                    }
                }
                Err(e) => {
                    let _ = tx.send(Msg::Fail(ChatError::Stream(e.to_string())));
                    return;
                }
            }
        }
    });

    let mut reply = Reply::default();
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(ChatError::Cancelled);
        }
        let msg = match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(m) => m,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(reply),
        };
        let line = match msg {
            Msg::Line(l) => l,
            Msg::Fail(e) => return Err(e),
            Msg::Eof => return Ok(reply),
        };
        let v: Value = match serde_json::from_str(line.trim()) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(err) = v.get("error").and_then(Value::as_str) {
            return Err(ChatError::Model(err.to_string()));
        }
        let m = &v["message"];
        if let Some(t) = m["thinking"].as_str().filter(|t| !t.is_empty()) {
            reply.thinking.push_str(t);
            on_delta(Delta::Thinking);
        }
        if let Some(c) = m["content"].as_str().filter(|c| !c.is_empty()) {
            reply.content.push_str(c);
            on_delta(Delta::Content(c));
        }
        if let Some(calls) = m["tool_calls"].as_array() {
            reply.tool_calls.extend(calls.iter().cloned());
        }
        if v["done"].as_bool().unwrap_or(false) {
            return Ok(reply);
        }
    }
}

// ---------- read-only endpoints ----------

fn get(base_url: &str, path: &str) -> Result<Value, String> {
    let url = format!("{}{path}", base_url.trim_end_matches('/'));
    agent(Duration::from_secs(5))
        .get(&url)
        .call()
        .map_err(|e| e.to_string())?
        .into_json()
        .map_err(|e| e.to_string())
}

pub fn version(base_url: &str) -> Result<String, String> {
    Ok(get(base_url, "/api/version")?["version"]
        .as_str()
        .unwrap_or("?")
        .to_string())
}

/// Names of every pulled model.
pub fn tags(base_url: &str) -> Result<Vec<String>, String> {
    let v = get(base_url, "/api/tags")?;
    Ok(v["models"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|m| m["name"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default())
}

/// `capabilities` from `/api/show`, e.g. ["completion", "tools", "thinking"].
pub fn capabilities(base_url: &str, model: &str) -> Result<Vec<String>, String> {
    let url = format!("{}/api/show", base_url.trim_end_matches('/'));
    let v: Value = agent(Duration::from_secs(10))
        .post(&url)
        .send_json(json!({ "model": model }))
        .map_err(|e| e.to_string())?
        .into_json()
        .map_err(|e| e.to_string())?;
    Ok(v["capabilities"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|c| c.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default())
}

#[allow(dead_code)] // read by `amadeus doctor`, added in the next commit
#[derive(Debug)]
pub struct Loaded {
    pub name: String,
    pub size: u64,
    pub size_vram: u64,
}

/// Models currently loaded, from `/api/ps`. `size_vram < size` means part of
/// the model spilled to CPU.
#[allow(dead_code)] // used by `amadeus doctor`, added in the next commit
pub fn ps(base_url: &str) -> Result<Vec<Loaded>, String> {
    let v = get(base_url, "/api/ps")?;
    Ok(v["models"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|m| Loaded {
                    name: m["name"].as_str().unwrap_or("?").to_string(),
                    size: m["size"].as_u64().unwrap_or(0),
                    size_vram: m["size_vram"].as_u64().unwrap_or(0),
                })
                .collect()
        })
        .unwrap_or_default())
}

/// True when `wanted` is pulled. A bare name matches its `:latest` tag.
pub fn is_pulled(pulled: &[String], wanted: &str) -> bool {
    pulled
        .iter()
        .any(|p| p == wanted || (!wanted.contains(':') && p == &format!("{wanted}:latest")))
}

/// Read one whole HTTP request (headers plus Content-Length body). Test
/// stubs must do this before replying: closing a socket with unread bytes
/// sends RST, and the client can lose the response.
#[cfg(test)]
pub fn read_request(sock: &mut impl std::io::Read) -> (String, Vec<u8>) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let (head_end, len) = loop {
        let n = sock.read(&mut chunk).unwrap_or(0);
        if n == 0 {
            return (String::from_utf8_lossy(&buf).into(), Vec::new());
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buf[..p]).to_lowercase();
            let len = head
                .lines()
                .find_map(|l| l.strip_prefix("content-length:"))
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(0);
            break (p + 4, len);
        }
    };
    while buf.len() < head_end + len {
        let n = sock.read(&mut chunk).unwrap_or(0);
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    (head, buf[head_end..].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::TcpListener;
    use std::time::Instant;

    /// Serve one canned HTTP response and close. Returns the port.
    fn stub(body: &'static str, status: &'static str) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            if let Ok((mut sock, _)) = listener.accept() {
                read_request(&mut sock);
                let resp = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/x-ndjson\r\n\
                     Connection: close\r\n\r\n{body}"
                );
                let _ = sock.write_all(resp.as_bytes());
            }
        });
        port
    }

    fn run(port: u16) -> (String, Result<Reply, ChatError>) {
        let cancel = Arc::new(AtomicBool::new(false));
        let mut text = String::new();
        let r = chat(
            &format!("http://127.0.0.1:{port}"),
            json!({"model": "m", "messages": []}),
            &cancel,
            |d| {
                if let Delta::Content(c) = d {
                    text.push_str(c)
                }
            },
        );
        (text, r)
    }

    #[test]
    fn content_streams_in_order() {
        let port = stub(
            "{\"message\":{\"content\":\"Hel\"},\"done\":false}\n\
             {\"message\":{\"content\":\"lo\"},\"done\":false}\n\
             {\"message\":{\"content\":\"\"},\"done\":true}\n",
            "200 OK",
        );
        let (text, r) = run(port);
        assert_eq!(text, "Hello");
        assert_eq!(r.unwrap().content, "Hello");
    }

    #[test]
    fn tool_calls_are_collected() {
        let port = stub(
            "{\"message\":{\"content\":\"\",\"tool_calls\":[{\"function\":{\"name\":\"list_dir\",\"arguments\":{\"path\":\".\"}}}]},\"done\":false}\n\
             {\"message\":{\"content\":\"\"},\"done\":true}\n",
            "200 OK",
        );
        let r = run(port).1.unwrap();
        assert_eq!(r.tool_calls.len(), 1);
        assert_eq!(r.tool_calls[0]["function"]["name"], "list_dir");
    }

    #[test]
    fn malformed_lines_are_skipped() {
        let port = stub(
            "not json\n{\"message\":{\"content\":\"ok\"},\"done\":true}\n",
            "200 OK",
        );
        assert_eq!(run(port).0, "ok");
    }

    #[test]
    fn error_object_is_a_model_error() {
        let port = stub("{\"error\":\"model not found\"}\n", "200 OK");
        assert_eq!(
            run(port).1.unwrap_err(),
            ChatError::Model("model not found".into())
        );
    }

    #[test]
    fn http_status_carries_ollamas_message() {
        let port = stub("{\"error\":\"model 'x' not found\"}", "404 Not Found");
        match run(port).1.unwrap_err() {
            ChatError::Status(404, d) => assert_eq!(d, "model 'x' not found"),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn unreachable_ollama_says_how_to_fix_it() {
        let cancel = Arc::new(AtomicBool::new(false));
        let e = chat("http://127.0.0.1:1", json!({}), &cancel, |_| {}).unwrap_err();
        assert!(e.to_string().contains("ollama serve"), "{e}");
    }

    #[test]
    fn cancel_returns_while_the_server_is_silent() {
        // Accept and never answer: what a cold model load looks like.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            let held = listener.accept();
            thread::sleep(Duration::from_secs(30));
            drop(held);
        });
        let cancel = Arc::new(AtomicBool::new(false));
        let c2 = cancel.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(200));
            c2.store(true, Ordering::Relaxed);
        });
        let t = Instant::now();
        let r = chat(
            &format!("http://127.0.0.1:{port}"),
            json!({}),
            &cancel,
            |_| {},
        );
        assert_eq!(r.unwrap_err(), ChatError::Cancelled);
        assert!(
            t.elapsed() < Duration::from_secs(2),
            "took {:?}",
            t.elapsed()
        );
    }

    #[test]
    fn bare_model_name_matches_latest() {
        let pulled = vec!["qwen3:8b".to_string(), "llama3.2:latest".to_string()];
        assert!(is_pulled(&pulled, "qwen3:8b"));
        assert!(is_pulled(&pulled, "llama3.2"));
        assert!(!is_pulled(&pulled, "qwen3:4b"));
    }
}
