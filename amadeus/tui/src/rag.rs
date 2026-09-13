use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::Sender;
use std::thread;

use crate::event::AppEvent;

/// Return the AMADEUS installation directory.
///
/// `AMADEUS_HOME` is preferred because the TUI can be launched from any
/// project directory.
pub fn amadeus_home() -> PathBuf {
    if let Ok(home) = std::env::var("AMADEUS_HOME") {
        return PathBuf::from(home);
    }

    // When installed through ~/.local/bin/amadeus, AMADEUS_HOME should
    // normally be set by amadeus.sh. This fallback is useful when running
    // the TUI binary directly from the AMADEUS checkout.
    if let Ok(exe) = std::env::current_exe() {
        let mut path = exe;

        // .../amadeus/tui/target/release/amadeus-tui
        for _ in 0..4 {
            if let Some(parent) = path.parent() {
                path = parent.to_path_buf();

                if path.join("rag/rag.py").is_file() && path.join("agent/agent.py").is_file() {
                    return path;
                }
            }
        }
    }

    PathBuf::from(".")
}

/// Python interpreter used by AMADEUS itself.
///
/// This is intentionally NOT rooted in the target project's venv.
pub fn python() -> PathBuf {
    let home = amadeus_home();
    let venv_python = home.join("venv/bin/python3");

    if venv_python.is_file() {
        venv_python
    } else {
        PathBuf::from("python3")
    }
}

/// AMADEUS's RAG script.
pub fn script() -> PathBuf {
    amadeus_home().join("rag/rag.py")
}

pub struct Retrieved {
    /// Candidate chunks with their cosine distances.
    pub hits: Vec<String>,

    /// Grounded prompt ready to send to Ollama.
    pub prompt: String,
}

pub enum QueryOutcome {
    Ok(Retrieved),
    NoContext,
    Err(String),
}

/// Run `rag.py query <question> --debug`.
pub fn query(root: &Path, question: &str) -> QueryOutcome {
    let mut command = Command::new(python());

    command
        .arg(script())
        .arg("query")
        .arg(question)
        .arg("--debug")
        .current_dir(root);

    let out = match command.output() {
        Ok(o) => o,
        Err(e) => {
            return QueryOutcome::Err(format!("cannot run rag.py: {e}"));
        }
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
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect();

    QueryOutcome::Ok(Retrieved {
        hits,
        prompt: stdout,
    })
}

/// Run `rag.py index <dir>`.
pub fn index(root: &Path, dir: &str, id: u64, tx: &Sender<AppEvent>) {
    let mut command = Command::new(python());

    command
        .arg(script())
        .arg("index")
        .arg(dir)
        .current_dir(root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(e) => {
            let _ = tx.send(AppEvent::Failed {
                id,
                text: format!("cannot run rag.py: {e}"),
            });
            return;
        }
    };

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let tx_stdout = tx.clone();

    let stdout_thread = thread::spawn(move || {
        if let Some(stdout) = stdout {
            for line in BufReader::new(stdout).lines() {
                match line {
                    Ok(line) if !line.trim().is_empty() => {
                        let _ = tx_stdout.send(AppEvent::Note {
                            id,
                            text: line,
                        });
                    }

                    Ok(_) => {}

                    Err(e) => {
                        let _ = tx_stdout.send(AppEvent::Note {
                            id,
                            text: format!("index stdout error: {e}"),
                        });
                        break;
                    }
                }
            }
        }
    });

    let tx_stderr = tx.clone();

    let stderr_thread = thread::spawn(move || {
        if let Some(stderr) = stderr {
            for line in BufReader::new(stderr).lines() {
                match line {
                    Ok(line) if !line.trim().is_empty() => {
                        let _ = tx_stderr.send(AppEvent::Note {
                            id,
                            text: line,
                        });
                    }

                    Ok(_) => {}

                    Err(e) => {
                        let _ = tx_stderr.send(AppEvent::Note {
                            id,
                            text: format!("index stderr error: {e}"),
                        });
                        break;
                    }
                }
            }
        }
    });

    let status = child.wait();

    let _ = stdout_thread.join();
    let _ = stderr_thread.join();

    match status {
        Ok(status) if status.success() => {
            let _ = tx.send(AppEvent::Done { id });
        }

        Ok(status) => {
            let _ = tx.send(AppEvent::Failed {
                id,
                text: format!("index exited with {status}"),
            });
        }

        Err(e) => {
            let _ = tx.send(AppEvent::Failed {
                id,
                text: format!("failed waiting for index: {e}"),
            });
        }
    }
}

pub fn shell(cmd: &str, id: u64, tx: &Sender<AppEvent>) {
    let output = match Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .output()
    {
        Ok(output) => output,
        Err(e) => {
            let _ = tx.send(AppEvent::Failed {
                id,
                text: format!("shell error: {e}"),
            });
            return;
        }
    };

    if !output.stdout.is_empty() {
        let text = String::from_utf8_lossy(&output.stdout);

        for line in text.lines() {
            if !line.trim().is_empty() {
                let _ = tx.send(AppEvent::Note {
                    id,
                    text: line.to_string(),
                });
            }
        }
    }

    if !output.stderr.is_empty() {
        let text = String::from_utf8_lossy(&output.stderr);

        for line in text.lines() {
            if !line.trim().is_empty() {
                let _ = tx.send(AppEvent::Note {
                    id,
                    text: line.to_string(),
                });
            }
        }
    }

    if output.status.success() {
        let _ = tx.send(AppEvent::Done { id });
    } else {
        let _ = tx.send(AppEvent::Failed {
            id,
            text: format!("shell exited with {}", output.status),
        });
    }
}
