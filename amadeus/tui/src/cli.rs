//! Non-interactive entry points: `amadeus-tui ask|rag …`. They share the
//! TUI's Ollama client, so think handling and errors are identical.

use std::io::Write;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use serde_json::json;

use crate::app::{chat_body, Config, Tier};
use crate::ollama::{self, Delta};
use crate::rag::{self, QueryOutcome};

fn tier_flag(args: &mut Vec<String>, default: Tier) -> Result<(Tier, bool), String> {
    let (mut tier, mut debug) = (default, false);
    while let Some(a) = args.first().filter(|a| a.starts_with("--")).cloned() {
        match a.as_str() {
            "--fast" => tier = Tier::Fast,
            "--main" => tier = Tier::Main,
            "--heavy" => tier = Tier::Heavy,
            "--debug" => debug = true,
            other => return Err(format!("unknown flag: {other}")),
        }
        args.remove(0);
    }
    Ok((tier, debug))
}

fn stream(cfg: &Config, model: &str, prompt: String) -> i32 {
    let cancel = Arc::new(AtomicBool::new(false));
    let body = chat_body(
        &cfg.ollama_url,
        model,
        vec![json!({"role": "user", "content": prompt})],
        None,
    );
    let mut out = std::io::stdout();
    let res = ollama::chat(&cfg.ollama_url, body, &cancel, |d| {
        if let Delta::Content(c) = d {
            let _ = out.write_all(c.as_bytes());
            let _ = out.flush();
        }
    });
    println!();
    match res {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    }
}

pub fn ask(mut args: Vec<String>) -> i32 {
    let cfg = Config::from_env();
    let (tier, _) = match tier_flag(&mut args, Tier::Main) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    if args.is_empty() {
        eprintln!("usage: amadeus ask [--fast|--heavy] <task>");
        return 2;
    }
    stream(&cfg, cfg.model(tier), args.join(" "))
}

pub fn rag(mut args: Vec<String>) -> i32 {
    let cfg = Config::from_env();
    let (tier, debug) = match tier_flag(&mut args, Tier::Main) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    if args.is_empty() {
        eprintln!("usage: amadeus rag [--debug] [--heavy] <question>");
        return 2;
    }
    let question = args.join(" ");
    let cancel = AtomicBool::new(false);
    match rag::query(&cfg.root, &question, &cancel) {
        QueryOutcome::Ok(r) => {
            if debug {
                // Retrieval only: the calibration view for the margin.
                for c in &r.candidates {
                    println!("{c}");
                }
                return 0;
            }
            eprintln!("sources: {}", r.sources.join(", "));
            stream(&cfg, cfg.model(tier), r.prompt)
        }
        QueryOutcome::NoContext => {
            println!("Nothing relevant in the index for that question.");
            0
        }
        QueryOutcome::Cancelled => 130,
        QueryOutcome::Err(e) => {
            eprintln!("error: {e}");
            1
        }
    }
}
