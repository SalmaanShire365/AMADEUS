//! `amadeus doctor`: is this machine ready to run AMADEUS? Every problem is
//! printed with the exact command that fixes it.
//!
//! By default it also loads the main model with a one-token request. That
//! makes `/api/ps` show where the model landed (GPU or CPU), and warms the
//! model so the first task in a demo doesn't pay the cold-load cost.

use std::fs;
use std::io::IsTerminal;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant, SystemTime};

use serde_json::json;

use crate::app::Config;
use crate::ollama;
use crate::paths;
use crate::templates;

const EMBED_MODEL: &str = "unclemusclez/jina-embeddings-v2-base-code";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Level {
    Ok,
    Warn,
    Fail,
}

struct Report {
    rows: Vec<(Level, String, Option<String>)>,
}

impl Report {
    fn add(&mut self, l: Level, msg: impl Into<String>, fix: Option<String>) {
        let (mark, color) = match l {
            Level::Ok => ("✓", "\x1b[32m"),
            Level::Warn => ("!", "\x1b[35m"),
            Level::Fail => ("✗", "\x1b[31m"),
        };
        let (color, reset) = if std::io::stdout().is_terminal() {
            (color, "\x1b[0m")
        } else {
            ("", "")
        };
        let msg = msg.into();
        println!("  {color}{mark}{reset} {msg}");
        if let Some(f) = &fix {
            println!("      fix: {f}");
        }
        self.rows.push((l, msg, fix));
    }

    fn failed(&self) -> usize {
        self.rows.iter().filter(|r| r.0 == Level::Fail).count()
    }

    fn warned(&self) -> usize {
        self.rows.iter().filter(|r| r.0 == Level::Warn).count()
    }
}

fn newest_mtime(dir: &Path) -> Option<SystemTime> {
    let mut newest = None;
    for e in fs::read_dir(dir).ok()?.flatten() {
        let p = e.path();
        let t = if p.is_dir() {
            newest_mtime(&p)
        } else {
            e.metadata().and_then(|m| m.modified()).ok()
        };
        if t > newest {
            newest = t;
        }
    }
    newest
}

fn gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1e9)
}

pub fn run(args: Vec<String>) -> i32 {
    let quick = args.iter().any(|a| a == "--quick");
    let cfg = Config::from_env();
    let home = paths::amadeus_home();
    let mut r = Report { rows: Vec::new() };
    println!("AMADEUS doctor · checkout {}", home.display());

    // --- Ollama ---
    println!("\nOllama ({})", cfg.ollama_url);
    let up = match ollama::version(&cfg.ollama_url) {
        Ok(v) => {
            r.add(Level::Ok, format!("server {v} is answering"), None);
            true
        }
        Err(e) => {
            let fix = if Command::new("ollama").arg("--version").output().is_ok() {
                "ollama serve   (or start the Ollama app)".to_string()
            } else {
                "install Ollama from https://ollama.com/download, then: ollama serve".to_string()
            };
            r.add(Level::Fail, format!("not reachable: {e}"), Some(fix));
            false
        }
    };

    if up {
        let pulled = ollama::tags(&cfg.ollama_url).unwrap_or_default();
        for (tier, name) in cfg.tiers() {
            let have = ollama::is_pulled(&pulled, &name);
            let fix = (!have).then(|| format!("ollama pull {name}"));
            match (tier, have) {
                (_, true) => r.add(Level::Ok, format!("{tier} model {name} is pulled"), None),
                ("main", false) => r.add(
                    Level::Fail,
                    format!("main model {name} is missing; every template uses it"),
                    fix,
                ),
                _ => r.add(
                    Level::Warn,
                    format!("{tier} model {name} is missing (only needed if you switch to it with ctrl+t)"),
                    fix,
                ),
            }
        }
        if ollama::is_pulled(&pulled, EMBED_MODEL) {
            r.add(Level::Ok, "embedding model for /rag is pulled", None);
        } else {
            r.add(
                Level::Warn,
                "embedding model for /rag and code search is missing (the other templates don't need it)",
                Some(format!("ollama pull {EMBED_MODEL}")),
            );
        }

        // --- GPU ---
        println!("\nGPU");
        let main = cfg.main_model.clone();
        if !quick && ollama::is_pulled(&pulled, &main) {
            let t = Instant::now();
            let warm = ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(300))
                .build()
                .post(&format!(
                    "{}/api/generate",
                    cfg.ollama_url.trim_end_matches('/')
                ))
                .send_json(json!({
                    "model": main, "prompt": "ok", "stream": false,
                    "options": {"num_predict": 1, "num_ctx": crate::agent::num_ctx()},
                    "keep_alive": "30m",
                }));
            match warm {
                Ok(_) => r.add(
                    Level::Ok,
                    format!(
                        "loaded {main} in {:.1}s (kept warm for 30 minutes)",
                        t.elapsed().as_secs_f32()
                    ),
                    None,
                ),
                Err(e) => r.add(
                    Level::Fail,
                    format!("couldn't load {main}: {e}"),
                    Some(format!("ollama run {main} 'hi'")),
                ),
            }
        }
        match ollama::ps(&cfg.ollama_url) {
            Ok(loaded) if loaded.is_empty() => r.add(
                Level::Warn,
                "no model loaded, so GPU placement is unknown",
                Some("amadeus doctor   (without --quick, which loads the main model)".into()),
            ),
            Ok(loaded) => {
                let ours: Vec<String> = cfg.tiers().into_iter().map(|(_, n)| n).collect();
                for m in loaded {
                    if !ollama::is_pulled(&ours, &m.name) {
                        r.add(
                            Level::Warn,
                            format!(
                                "{} ({}) is also loaded, probably by another program; it shares the GPU",
                                m.name,
                                gb(m.size)
                            ),
                            Some(format!("ollama stop {}   (before a demo)", m.name)),
                        );
                        continue;
                    }
                    let pct = (m.size_vram * 100).checked_div(m.size).unwrap_or(0);
                    let what = format!("{} uses {} ({pct}% on GPU)", m.name, gb(m.size));
                    if pct >= 95 {
                        r.add(Level::Ok, what, None);
                    } else if pct == 0 {
                        r.add(
                            Level::Warn,
                            format!("{what}: running on CPU only, expect slow answers"),
                            Some("check your GPU driver is visible to Ollama (nvidia-smi), or use a smaller model: AMADEUS_MAIN_MODEL=qwen3:4b-instruct".into()),
                        );
                    } else {
                        r.add(
                            Level::Warn,
                            format!("{what}: part runs on CPU, answers will be slower"),
                            Some("free GPU memory, lower AMADEUS_NUM_CTX, or use AMADEUS_MAIN_MODEL=qwen3:4b-instruct".into()),
                        );
                    }
                }
            }
            Err(e) => r.add(Level::Warn, format!("couldn't read /api/ps: {e}"), None),
        }
    }

    // --- AMADEUS itself ---
    println!("\nAMADEUS");
    let (tpls, errors) = templates::load_all();
    if tpls.is_empty() {
        r.add(
            Level::Fail,
            "no templates found",
            Some(format!("check {}", home.join("templates").display())),
        );
    } else {
        let names: Vec<&str> = tpls.iter().map(|t| t.name.as_str()).collect();
        r.add(
            Level::Ok,
            format!("{} templates: {}", tpls.len(), names.join(", ")),
            None,
        );
    }
    for e in errors {
        r.add(Level::Warn, format!("template skipped: {e}"), None);
    }

    let exe_time = std::env::current_exe()
        .ok()
        .and_then(|p| fs::metadata(p).ok())
        .and_then(|m| m.modified().ok());
    match (exe_time, newest_mtime(&home.join("tui/src"))) {
        (Some(b), Some(s)) if s > b => r.add(
            Level::Warn,
            "TUI binary is older than its sources",
            Some(format!(
                "cd {} && cargo build --release",
                home.join("tui").display()
            )),
        ),
        _ => r.add(Level::Ok, "TUI build is up to date", None),
    }

    let py = home.join("venv/bin/python3");
    let venv_ok = py.is_file()
        && Command::new(&py)
            .args(["-c", "import sqlite_vec"])
            .output()
            .is_ok_and(|o| o.status.success());
    if venv_ok {
        r.add(Level::Ok, "Python venv for /rag is ready", None);
    } else {
        r.add(
            Level::Warn,
            "Python venv for /rag is missing (only code search needs it)",
            Some(format!(
                "cd {} && make setup",
                home.parent().unwrap_or(&home).display()
            )),
        );
    }

    let state = paths::state_dir();
    let writable =
        fs::create_dir_all(&state).is_ok() && fs::write(state.join(".doctor"), b"ok").is_ok();
    let _ = fs::remove_file(state.join(".doctor"));
    if writable {
        r.add(
            Level::Ok,
            format!("run logs go to {}", state.display()),
            None,
        );
    } else {
        r.add(
            Level::Fail,
            format!("can't write {}", state.display()),
            Some("set XDG_STATE_HOME to a writable directory".into()),
        );
    }

    println!();
    match (r.failed(), r.warned()) {
        (0, 0) => println!("Ready."),
        (0, w) => println!("Ready, with {w} optional item(s) above."),
        (f, _) => println!("{f} problem(s) to fix before AMADEUS can run."),
    }
    if r.failed() > 0 {
        1
    } else {
        0
    }
}
