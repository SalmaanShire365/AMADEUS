//! Thin wrapper over rag.py. Retrieval behaviour lives entirely in Python;
//! this only spawns it (killably) and reads its output.

use std::path::Path;
use std::process::Command;
use std::sync::atomic::AtomicBool;

use crate::paths;
use crate::proc::{self, Exit};

pub struct Retrieved {
    /// Grounded prompt ready to send to the model.
    pub prompt: String,
    /// Chunks that made it into the prompt, as `path:start-end`.
    pub sources: Vec<String>,
    /// Raw top-k candidates with distances, before the margin cut. Shown only
    /// with /debug.
    pub candidates: Vec<String>,
}

pub enum QueryOutcome {
    Ok(Retrieved),
    NoContext,
    Cancelled,
    Err(String),
}

fn command(root: &Path) -> Command {
    let mut c = Command::new(paths::python());
    c.arg(paths::rag_script()).current_dir(root);
    c
}

/// The `### path (lines a-b)` headers rag.py writes for each chunk it kept.
pub fn parse_sources(prompt: &str) -> Vec<String> {
    prompt
        .lines()
        .filter_map(|l| {
            let rest = l.strip_prefix("### ")?;
            let (path, range) = rest.rsplit_once(" (lines ")?;
            let range = range.strip_suffix(')')?;
            Some(format!("{path}:{range}"))
        })
        .collect()
}

pub fn query(root: &Path, question: &str, cancel: &AtomicBool) -> QueryOutcome {
    let mut c = command(root);
    c.arg("query").arg(question).arg("--debug");
    let out = match proc::run(c, cancel, None, |_, _| {}) {
        Ok(o) => o,
        Err(e) => return QueryOutcome::Err(format!("cannot run rag.py: {e}")),
    };
    match out.exit {
        Exit::Cancelled => return QueryOutcome::Cancelled,
        Exit::TimedOut => return QueryOutcome::Err("rag.py timed out".into()),
        Exit::Status(s) if !s.success() => {
            let msg = out.stderr.trim();
            return QueryOutcome::Err(if msg.is_empty() {
                format!("rag.py exited {s}")
            } else {
                msg.lines().last().unwrap_or(msg).to_string()
            });
        }
        Exit::Status(_) => {}
    }
    let stdout = out.stdout.trim();
    if stdout == "NO_RELEVANT_CONTEXT" {
        return QueryOutcome::NoContext;
    }
    if stdout.is_empty() {
        return QueryOutcome::Err("rag.py produced no prompt".into());
    }
    QueryOutcome::Ok(Retrieved {
        sources: parse_sources(stdout),
        candidates: out
            .stderr
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect(),
        prompt: stdout.to_string(),
    })
}

/// `rag.py index <dir>`, streaming progress lines to `on_line`.
pub fn index(
    root: &Path,
    dir: &str,
    cancel: &AtomicBool,
    mut on_line: impl FnMut(&str),
) -> Result<(), String> {
    let mut c = command(root);
    c.arg("index").arg(dir);
    let out = proc::run(c, cancel, None, |_, l| {
        if !l.trim().is_empty() {
            on_line(l)
        }
    })
    .map_err(|e| format!("cannot run rag.py: {e}"))?;
    if out.success() {
        Ok(())
    } else {
        Err(format!("index {}", out.describe()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sources_come_from_the_prompt_headers() {
        let prompt = "You are AMADEUS.\n--- Context ---\n### a/b.go (lines 10-40)\n```\nx\n```\n\n\
                      ### c (d).go (lines 1-5)\n```\ny\n```\n";
        assert_eq!(parse_sources(prompt), vec!["a/b.go:10-40", "c (d).go:1-5"]);
    }
}
