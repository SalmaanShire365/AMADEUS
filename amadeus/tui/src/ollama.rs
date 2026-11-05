//! Ollama HTTP client: streaming `/api/chat` plus the small read-only
//! endpoints the health check and doctor use.
//!
//! Cancel must stop generation on the server, not just stop listening. Two
//! things make that hard. A cold model load sends no bytes for tens of
//! seconds. And while a model writes a tool call, Ollama buffers it and
//! sends nothing until the call is complete. A reader blocked in `read()`
//! can't see a cancel flag in either case. So `/api/chat` uses a small
//! HTTP/1.1 client over a `TcpStream` we keep a handle to. On cancel the
//! socket is shut down, the blocked read returns at once, and Ollama sees
//! the disconnect and stops. Measured: without this, a cancelled runaway
//! turn kept the GPU busy and queued the next request behind it.

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
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
    /// Generation stopped at `num_predict`, not because the model finished.
    pub truncated: bool,
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

/// `http://host:port/prefix` → ("host:port", "/prefix").
fn split_url(base: &str) -> Result<(String, String), String> {
    let rest = base
        .strip_prefix("http://")
        .ok_or_else(|| format!("{base}: only http:// URLs are supported for Ollama"))?;
    let (host, path) = match rest.find('/') {
        Some(i) => (&rest[..i], rest[i..].trim_end_matches('/')),
        None => (rest, ""),
    };
    let host = if host.contains(':') {
        host.to_string()
    } else {
        format!("{host}:80")
    };
    Ok((host, path.to_string()))
}

/// Decodes `Transfer-Encoding: chunked`. Any error is fatal for the stream,
/// so there's no partial state to resume.
struct Chunked<R> {
    inner: R,
    left: usize,
    done: bool,
}

impl<R: BufRead> Read for Chunked<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.done {
            return Ok(0);
        }
        if self.left == 0 {
            let mut size = String::new();
            if self.inner.read_line(&mut size)? == 0 {
                return Ok(0);
            }
            let hex = size.trim().split(';').next().unwrap_or("");
            self.left = usize::from_str_radix(hex, 16)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad chunk size"))?;
            if self.left == 0 {
                self.done = true;
                return Ok(0);
            }
        }
        let want = buf.len().min(self.left);
        let n = self.inner.read(&mut buf[..want])?;
        if n == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        self.left -= n;
        if self.left == 0 {
            let mut crlf = String::new();
            self.inner.read_line(&mut crlf)?;
        }
        Ok(n)
    }
}

/// POST `body` and return the response body as a line reader, with the
/// socket stored in `handle` so another thread can shut it down.
fn post_stream(
    base: &str,
    path: &str,
    body: &[u8],
    stall: Duration,
    handle: &Mutex<Option<TcpStream>>,
) -> Result<Box<dyn BufRead + Send>, ChatError> {
    let (host, prefix) = split_url(base).map_err(ChatError::Unreachable)?;
    let addr = host
        .to_socket_addrs()
        .map_err(|e| ChatError::Unreachable(e.to_string()))?
        .next()
        .ok_or_else(|| ChatError::Unreachable(format!("{host}: no address")))?;
    let mut sock = TcpStream::connect_timeout(&addr, Duration::from_secs(5))
        .map_err(|e| ChatError::Unreachable(e.to_string()))?;
    let unreachable = |e: io::Error| ChatError::Unreachable(e.to_string());
    sock.set_read_timeout(Some(stall)).map_err(unreachable)?;
    *handle.lock().unwrap() = Some(sock.try_clone().map_err(unreachable)?);

    let head = format!(
        "POST {prefix}{path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    sock.write_all(head.as_bytes())
        .and_then(|_| sock.write_all(body))
        .map_err(unreachable)?;

    let mut r = BufReader::new(sock);
    let stream_err = |e: io::Error| ChatError::Stream(e.to_string());
    let mut status = String::new();
    r.read_line(&mut status).map_err(stream_err)?;
    let code: u16 = status
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse().ok())
        .ok_or_else(|| ChatError::Stream(format!("bad status line {status:?}")))?;
    let mut chunked = false;
    loop {
        let mut h = String::new();
        if r.read_line(&mut h).map_err(stream_err)? == 0 || h.trim().is_empty() {
            break;
        }
        let h = h.to_ascii_lowercase();
        if h.starts_with("transfer-encoding:") && h.contains("chunked") {
            chunked = true;
        }
    }
    let mut body: Box<dyn BufRead + Send> = if chunked {
        Box::new(BufReader::new(Chunked {
            inner: r,
            left: 0,
            done: false,
        }))
    } else {
        Box::new(r)
    };
    if code != 200 {
        let mut text = String::new();
        let _ = body.read_to_string(&mut text);
        let detail = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|v| v["error"].as_str().map(str::to_string))
            .unwrap_or(text);
        return Err(ChatError::Status(code, detail.trim().into()));
    }
    Ok(body)
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
    let base = base_url.trim_end_matches('/').to_string();
    let (tx, rx) = mpsc::channel::<Msg>();
    let handle: Arc<Mutex<Option<TcpStream>>> = Arc::new(Mutex::new(None));
    let h2 = handle.clone();
    let flag = cancel.clone();
    let stall = stall_timeout();

    thread::spawn(move || {
        let bytes = body.to_string().into_bytes();
        let mut reader = match post_stream(&base, "/api/chat", &bytes, stall, &h2) {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.send(Msg::Fail(e));
                return;
            }
        };
        let mut line = String::new();
        loop {
            if flag.load(Ordering::Relaxed) {
                return;
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

    let hang_up = || {
        if let Some(s) = handle.lock().unwrap().take() {
            let _ = s.shutdown(Shutdown::Both);
        }
    };
    let mut reply = Reply::default();
    loop {
        if cancel.load(Ordering::Relaxed) {
            hang_up();
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
            hang_up();
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
            reply.truncated = v["done_reason"] == "length";
            hang_up();
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

#[derive(Debug)]
pub struct Loaded {
    pub name: String,
    pub size: u64,
    pub size_vram: u64,
}

/// Models currently loaded, from `/api/ps`. `size_vram < size` means part of
/// the model spilled to CPU.
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
    fn chunked_bodies_are_decoded() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            if let Ok((mut sock, _)) = listener.accept() {
                read_request(&mut sock);
                let a = "{\"message\":{\"content\":\"Hel\"}}\n";
                let b = "{\"message\":{\"content\":\"lo\"},\"done\":true}\n";
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{a}\r\n{:x}\r\n{b}\r\n0\r\n\r\n",
                    a.len(),
                    b.len()
                );
                let _ = sock.write_all(resp.as_bytes());
            }
        });
        let (text, r) = run(port);
        assert_eq!(text, "Hello");
        assert!(r.is_ok());
    }

    #[test]
    fn cancel_hangs_up_so_the_server_stops() {
        // A server that streams nothing (a tool call being buffered) and
        // reports when the client disconnects.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (gone_tx, gone_rx) = mpsc::channel();
        thread::spawn(move || {
            if let Ok((mut sock, _)) = listener.accept() {
                read_request(&mut sock);
                let _ = sock.write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n");
                let mut buf = [0u8; 1];
                // Blocks until the client closes its end.
                let _ = sock.read(&mut buf);
                let _ = gone_tx.send(());
            }
        });
        let cancel = Arc::new(AtomicBool::new(false));
        let c2 = cancel.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(200));
            c2.store(true, Ordering::Relaxed);
        });
        let r = chat(
            &format!("http://127.0.0.1:{port}"),
            json!({}),
            &cancel,
            |_| {},
        );
        assert_eq!(r.unwrap_err(), ChatError::Cancelled);
        assert!(
            gone_rx.recv_timeout(Duration::from_secs(2)).is_ok(),
            "server never saw the disconnect"
        );
    }

    #[test]
    fn urls_split_into_host_and_prefix() {
        assert_eq!(
            split_url("http://localhost:11434").unwrap(),
            ("localhost:11434".into(), "".into())
        );
        assert_eq!(
            split_url("http://box/ollama/").unwrap(),
            ("box:80".into(), "/ollama".into())
        );
        assert!(split_url("https://x").is_err());
    }

    #[test]
    fn bare_model_name_matches_latest() {
        let pulled = vec!["qwen3:8b".to_string(), "llama3.2:latest".to_string()];
        assert!(is_pulled(&pulled, "qwen3:8b"));
        assert!(is_pulled(&pulled, "llama3.2"));
        assert!(!is_pulled(&pulled, "qwen3:4b"));
    }
}
