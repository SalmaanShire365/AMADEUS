//! One JSONL file per task: every model turn, tool call, approval and
//! result, so a run can be replayed and audited afterwards.
//!
//! Lives under $XDG_STATE_HOME/amadeus/projects/<slug>/runs/, never in the
//! workspace.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

pub struct RunLog {
    pub path: PathBuf,
    file: Option<File>,
}

fn now() -> (u64, String) {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    (d.as_secs(), iso8601(d.as_secs()))
}

/// UTC `YYYY-MM-DDTHH:MM:SSZ` from Unix seconds (Howard Hinnant's
/// civil-from-days algorithm), so there's no date crate for one format.
pub fn iso8601(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem / 60) % 60,
        rem % 60
    )
}

/// `YYYY-MM-DD` (UTC), for prompts that need to know the year.
pub fn today() -> String {
    now().1[..10].to_string()
}

impl RunLog {
    /// Open `<dir>/<timestamp>-<template>.jsonl`. Logging failures never stop
    /// a run; the log just stays empty and `path` says where it would be.
    pub fn create(dir: &Path, template: &str) -> RunLog {
        let (_, ts) = now();
        let stamp = ts.replace([':', '-'], "").trim_end_matches('Z').to_string();
        let path = dir.join(format!("{stamp}-{template}.jsonl"));
        let _ = fs::create_dir_all(dir);
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .ok();
        RunLog { path, file }
    }

    pub fn log(&mut self, kind: &str, mut data: Value) {
        let (secs, ts) = now();
        if !data.is_object() {
            data = json!({ "value": data });
        }
        data["ts"] = json!(ts);
        data["t"] = json!(secs);
        data["kind"] = json!(kind);
        if let Some(f) = self.file.as_mut() {
            let _ = writeln!(f, "{data}");
        }
    }
}

/// Most recent run log in `dir`, by file name (names start with a UTC stamp).
pub fn latest(dir: &Path) -> Option<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
        .collect();
    files.sort();
    files.pop()
}

fn short(s: &str, n: usize) -> String {
    let one: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if one.chars().count() > n {
        format!("{}…", one.chars().take(n).collect::<String>())
    } else {
        one
    }
}

/// A human-readable transcript of a run log, for `amadeus replay`.
pub fn replay(path: &Path) -> Result<String, String> {
    let f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    for line in BufReader::new(f).lines().map_while(Result::ok) {
        let Ok(v) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let ts = v["ts"].as_str().unwrap_or("");
        let time = ts.get(11..19).unwrap_or(ts);
        let s = |k: &str| v[k].as_str().unwrap_or("").to_string();
        let row = match v["kind"].as_str().unwrap_or("") {
            "run_start" => format!(
                "{time}  START   {} on {} in {}{}\n          task: {}",
                s("template"),
                s("model"),
                s("workspace"),
                if v["auto_approve"].as_bool() == Some(true) {
                    " (auto-approve)"
                } else {
                    ""
                },
                s("task")
            ),
            "model_turn" => {
                let calls = v["tool_calls"].as_array().map_or(0, Vec::len);
                let what = if calls > 0 {
                    format!("{calls} tool call(s)")
                } else {
                    format!("answer: {}", short(&s("content"), 100))
                };
                format!(
                    "{time}  MODEL   step {} ({} ms): {what}",
                    v["step"], v["duration_ms"]
                )
            }
            "tool_call" => format!("{time}  CALL    {}", s("headline")),
            "approval" => format!(
                "{time}  {:<7} {}",
                s("decision").to_uppercase(),
                s("headline")
            ),
            "tool_result" => format!(
                "{time}  {}  {}",
                if v["ok"].as_bool() == Some(true) {
                    "RESULT "
                } else {
                    "FAILED "
                },
                short(&s("summary"), 100)
            ),
            "run_end" => format!("{time}  END     {} after {} steps", s("status"), v["steps"]),
            other => format!("{time}  {other}"),
        };
        out.push(row);
    }
    Ok(out.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_dates_match_known_values() {
        assert_eq!(iso8601(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso8601(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(iso8601(1_793_057_400), "2026-10-26T23:30:00Z");
    }

    #[test]
    fn log_lines_are_json_with_kind_and_time_and_replay_reads_them() {
        let dir = std::env::temp_dir().join(format!("amadeus-runlog-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut log = RunLog::create(&dir, "folder-summary");
        log.log(
            "run_start",
            json!({"template": "folder-summary", "model": "m", "task": "do it", "workspace": "/w"}),
        );
        log.log(
            "approval",
            json!({"decision": "approved", "headline": "write_file a.md"}),
        );
        log.log("run_end", json!({"status": "done", "steps": 3}));
        drop(log);

        let path = latest(&dir).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        let first: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        assert_eq!(first["kind"], "run_start");
        assert!(first["ts"].as_str().unwrap().ends_with('Z'));

        let r = replay(&path).unwrap();
        assert!(r.contains("START   folder-summary on m in /w"), "{r}");
        assert!(r.contains("APPROVED write_file a.md"), "{r}");
        assert!(r.contains("END     done after 3 steps"), "{r}");
        let _ = fs::remove_dir_all(dir);
    }
}
