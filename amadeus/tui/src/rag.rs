use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::Sender;

use crate::event::AppEvent;

/// Mirrors `rag_python()` in amadeus.sh: prefer the venv interpreter, fall
/// back to the system one.
pub fn python(root: &Path) -> PathBuf {
    let venv = root.join("rag/venv/bin/python3");
    if venv.is_file() {
        venv
    } else {
        PathBuf::from("python3")
    }
}

pub fn script(root: &Path) -> PathBuf {
    root.join("rag/rag.py")
}

pub struct Retrieved {
    /// Candidate chunks with their cosine distances, newest-first as printed.
    /// These come from `--debug` on stderr, which prints the raw top-k BEFORE
    /// reranking and before the DISTANCE_MARGIN cut — so some of these will
    /// not actually be in the prompt.
    pub hits: Vec<String>,
    /// The grounded prompt on stdout, ready to send to Ollama.
    pub prompt: String,
}

pub enum QueryOutcome {
    Ok(Retrieved),
    NoContext,
    Err(String),
}

/// Run `rag.py query <question> --debug`. One-shot, blocking; call it on a
/// worker thread. Costs a Python start plus a sqlite open (~200ms) — the
/// embedding model itself lives in Ollama, not here, so there is nothing
/// warm to keep.
pub fn query(root: &Path, question: &str) -> QueryOutcome {
    let out = Command::new(python(root))
        .arg(script(root))
        .arg("query")
        .arg(question)
        .arg("--debug")
        .output();

    let out = match out {
        Ok(o) => o,
        Err(e) => return QueryOutcome::Err(format!("cannot run rag.py: {e}")),
    };

    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();

    if !out.status.success() {
        let msg = if stderr.trim().is_empty() {
            format!("rag.py exited {}", out.status)
        } else {
            stderr.trim().to_string()
        };
        return QueryOutcome::Err(msg);
    }

    if stdout.trim() == "NO_RELEVANT_CONTEXT" {
        return QueryOutcome::NoContext;
    }
    if stdout.trim().is_empty() {
        return QueryOutcome::Err("rag.py produced no prompt".into());
    }

    let hits = stderr
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect();

    QueryOutcome::Ok(Retrieved { hits, prompt: stdout })
}

/// Run `rag.py index <dir>`, forwarding each line of progress as it appears.
pub fn index(root: &Path, dir: &str, id: u64, tx: &Sender<AppEvent>) {
    let child = Command::new(python(root))
        .arg(script(root))
        .arg("index")
        .arg(dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();

    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            let _ = tx.send(AppEvent::Failed { id, text: format!("cannot run rag.py: {e}") });
            return;
        }
    };

    if let Some(stdout) = child.stdout.take() {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if tx.send(AppEvent::Note { id, text: line }).is_err() {
                return;
            }
        }
    }

    let mut errs = String::new();
    if let Some(stderr) = child.stderr.take() {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            errs.push_str(&line);
            errs.push('\n');
        }
    }

    match child.wait() {
        Ok(st) if st.success() => {
            let _ = tx.send(AppEvent::Done { id });
        }
        Ok(st) => {
            let text = if errs.trim().is_empty() {
                format!("index exited {st}")
            } else {
                errs.trim().to_string()
            };
            let _ = tx.send(AppEvent::Failed { id, text });
        }
        Err(e) => {
            let _ = tx.send(AppEvent::Failed { id, text: e.to_string() });
        }
    }
}

/// Run a shell command with output captured into scrollback rather than
/// letting the child touch the terminal — we are in raw mode on the alternate
/// screen, so anything interactive would fight the UI.
pub fn shell(cmd: &str, id: u64, tx: &Sender<AppEvent>) {
    let out = Command::new("sh").arg("-c").arg(cmd).output();
    match out {
        Ok(o) => {
            let mut text = String::from_utf8_lossy(&o.stdout).to_string();
            text.push_str(&String::from_utf8_lossy(&o.stderr));
            if text.trim().is_empty() {
                text = "(no output)".into();
            }
            let _ = tx.send(AppEvent::Note { id, text: text.trim_end().to_string() });
            if !o.status.success() {
                let _ = tx.send(AppEvent::Note { id, text: format!("exit {}", o.status) });
            }
            let _ = tx.send(AppEvent::Done { id });
        }
        Err(e) => {
            let _ = tx.send(AppEvent::Failed { id, text: e.to_string() });
        }
    }
}