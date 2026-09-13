use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;

use crate::event::AppEvent;

/// Stream a completion from Ollama, emitting one `Token` event per chunk.
///
/// This is the whole reason the TUI exists: the shell version captured
/// `ollama run` with a subshell, which blocks until generation is complete.
/// `/api/generate` with `stream: true` gives us tokens as they are produced.
///
/// Always emits exactly one terminal event (`Done` or `Failed`) so the caller
/// can rely on `busy` being cleared.
pub fn generate(
    base_url: &str,
    model: &str,
    prompt: &str,
    id: u64,
    tx: &Sender<AppEvent>,
    cancel: &AtomicBool,
) {
    let url = format!("{}/api/generate", base_url.trim_end_matches('/'));
    let body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": true,
        "options": { "temperature": 0.2 },
    });

    let resp = match ureq::post(&url).send_json(body) {
        Ok(r) => r,
        Err(ureq::Error::Status(code, r)) => {
            let detail = r.into_string().unwrap_or_default();
            let _ = tx.send(AppEvent::Failed {
                id,
                text: format!("ollama returned {code}: {}", detail.trim()),
            });
            return;
        }
        Err(e) => {
            let _ = tx.send(AppEvent::Failed {
                id,
                text: format!(
                    "cannot reach ollama at {base_url} ({e}). is `ollama serve` running?"
                ),
            });
            return;
        }
    };

    let mut reader = BufReader::new(resp.into_reader());
    let mut line = String::new();
    loop {
        // Cancellation is checked between lines. A request that is stalled
        // with no bytes arriving will not notice until the next token or EOF.
        if cancel.load(Ordering::Relaxed) {
            let _ = tx.send(AppEvent::Note { id, text: "[cancelled]".into() });
            break;
        }
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                let _ = tx.send(AppEvent::Failed { id, text: format!("stream broke: {e}") });
                return;
            }
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
            let _ = tx.send(AppEvent::Failed { id, text: format!("ollama: {err}") });
            return;
        }
        if let Some(delta) = v.get("response").and_then(|r| r.as_str()) {
            if !delta.is_empty() {
                if tx.send(AppEvent::Token { id, delta: delta.to_string() }).is_err() {
                    return; // UI is gone
                }
            }
        }
        if v.get("done").and_then(|d| d.as_bool()).unwrap_or(false) {
            break;
        }
    }

    let _ = tx.send(AppEvent::Done { id });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;

    /// Serve one canned HTTP response and close. Returns the port.
    fn stub(body: &'static str, status: &'static str) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            if let Ok((mut sock, _)) = listener.accept() {
                let mut buf = [0u8; 4096];
                let _ = sock.read(&mut buf);
                let resp = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/x-ndjson\r\n\
                     Connection: close\r\n\r\n{body}"
                );
                let _ = sock.write_all(resp.as_bytes());
            }
        });
        port
    }

    fn collect(port: u16) -> (String, Vec<String>, bool) {
        let (tx, rx) = mpsc::channel();
        let cancel = AtomicBool::new(false);
        generate(&format!("http://127.0.0.1:{port}"), "m", "p", 7, &tx, &cancel);
        drop(tx);
        let (mut text, mut errs, mut done) = (String::new(), Vec::new(), false);
        while let Ok(ev) = rx.try_recv() {
            match ev {
                AppEvent::Token { id, delta } => {
                    assert_eq!(id, 7);
                    text.push_str(&delta);
                }
                AppEvent::Done { .. } => done = true,
                AppEvent::Failed { text: t, .. } => errs.push(t),
                _ => {}
            }
        }
        (text, errs, done)
    }

    #[test]
    fn tokens_arrive_in_order_and_done_ends_the_stream() {
        let port = stub(
            "{\"response\":\"Hel\",\"done\":false}\n\
             {\"response\":\"lo\",\"done\":false}\n\
             {\"response\":\"\",\"done\":true}\n",
            "200 OK",
        );
        let (text, errs, done) = collect(port);
        assert_eq!(text, "Hello");
        assert!(errs.is_empty(), "unexpected failures: {errs:?}");
        assert!(done);
    }

    #[test]
    fn malformed_lines_are_skipped_not_fatal() {
        let port = stub(
            "not json at all\n\
             {\"response\":\"ok\",\"done\":false}\n\
             \n\
             {\"done\":true}\n",
            "200 OK",
        );
        let (text, errs, done) = collect(port);
        assert_eq!(text, "ok");
        assert!(errs.is_empty(), "unexpected failures: {errs:?}");
        assert!(done);
    }

    #[test]
    fn an_error_object_ends_the_request_without_a_done() {
        let port = stub("{\"error\":\"model not found\"}\n", "200 OK");
        let (text, errs, done) = collect(port);
        assert!(text.is_empty());
        assert_eq!(errs.len(), 1);
        assert!(errs[0].contains("model not found"));
        assert!(!done, "Failed must be terminal, not followed by Done");
    }

    #[test]
    fn unreachable_ollama_reports_a_usable_message() {
        // Port 1 is reserved and will refuse.
        let (tx, rx) = mpsc::channel();
        let cancel = AtomicBool::new(false);
        generate("http://127.0.0.1:1", "m", "p", 7, &tx, &cancel);
        drop(tx);
        let ev = rx.try_recv().unwrap();
        match ev {
            AppEvent::Failed { text, .. } => assert!(text.contains("ollama serve")),
            other => panic!("expected Failed, got {other:?}"),
        }
    }
}