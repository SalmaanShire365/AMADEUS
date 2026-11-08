//! The agent loop. One implementation, used by every template.
//!
//! model turn (streamed) → tool calls → approval where the tool needs it →
//! execute → results back to the model → repeat, until the model answers
//! without calling a tool, the step budget runs out, or the user cancels.
//! Every step is written to the run log.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::app::chat_body;
use crate::approval::{Gate, Verdict};
use crate::event::AppEvent;
use crate::ollama::{self, ChatError, Delta};
use crate::render;
use crate::runlog::RunLog;
use crate::templates::Template;
use crate::tools::{self, Call, Workspace};

pub struct RunSpec {
    pub url: String,
    pub model: String,
    pub template: Template,
    pub root: PathBuf,
    /// Earlier turns of this conversation, system prompt first. Empty starts
    /// a fresh conversation.
    pub history: Vec<Value>,
    pub task: String,
    pub auto: bool,
    pub log_dir: PathBuf,
}

/// Context window requested from Ollama. 8k fits qwen3:8b plus its KV cache
/// in 8 GB of VRAM; raise it on bigger cards.
pub fn num_ctx() -> u64 {
    std::env::var("AMADEUS_NUM_CTX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8192)
}

/// Most tokens one model turn may generate. 3k covers a full CSV, summary or
/// drafted reply; it exists to stop runaway turns.
pub fn num_predict() -> u64 {
    std::env::var("AMADEUS_NUM_PREDICT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3072)
}

pub fn system_prompt(t: &Template, root: &std::path::Path) -> String {
    format!(
        "{}\n\n\
         You are running inside AMADEUS on the user's own computer. Nothing leaves this machine.\n\
         Today is {}. The workspace folder is {}. Every path you use is relative to it.\n\
         Look at files with the tools before you describe them. Never invent file contents.\n\
         If the user declines a tool call, do not repeat it. Continue without it, or say what you would have done.\n\
         When you are finished, reply with a short final answer in markdown and no tool call.",
        t.system.trim(),
        crate::runlog::today(),
        root.display()
    )
}

/// The `tool_calls` shape Ollama expects back in the assistant message.
fn native(calls: &[Call]) -> Value {
    Value::Array(
        calls
            .iter()
            .map(|c| json!({"function": {"name": c.name, "arguments": c.args}}))
            .collect(),
    )
}

/// The workspace path a file tool call names, normalised so `./a.txt` and
/// `a.txt` match.
fn read_key(call: &Call) -> Option<String> {
    let p = call.args["path"].as_str()?.trim();
    let p = p.trim_start_matches("./").trim_end_matches('/');
    (!p.is_empty()).then(|| p.to_string())
}

fn wait_for(rx: &mpsc::Receiver<Verdict>, cancel: &AtomicBool) -> Option<Verdict> {
    loop {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(d) => return Some(d),
            Err(RecvTimeoutError::Timeout) if !cancel.load(Ordering::Relaxed) => continue,
            Err(_) => return None,
        }
    }
}

/// How a side-effecting call came out of the approval gate.
pub(crate) enum Gated {
    Ran(tools::Outcome),
    /// The user said no, with their note if they gave one.
    Declined(Option<String>),
    /// The call couldn't be previewed (bad path, missing argument).
    Refused(String),
    Cancelled,
}

/// Ask where the gate says to, then run the call. Logs the approval and the
/// result, and reports the result to the UI. A decline is only logged: what
/// the UI and the model are told about it is up to the caller.
#[allow(clippy::too_many_arguments)]
pub(crate) fn approve_and_run(
    ws: &Workspace,
    gate: &mut Gate,
    call: &Call,
    needs_approval: bool,
    log: &mut RunLog,
    tx: &Sender<AppEvent>,
    id: u64,
    cancel: &AtomicBool,
) -> Gated {
    let headline = call.headline();
    if gate.must_ask(needs_approval) {
        let preview = match tools::preview(ws, call) {
            Ok(p) => p,
            Err(e) => {
                let _ = tx.send(AppEvent::Tool {
                    id,
                    headline,
                    summary: e.clone(),
                    ok: false,
                });
                log.log(
                    "tool_result",
                    json!({"name": call.name, "ok": false, "summary": e}),
                );
                return Gated::Refused(e);
            }
        };
        let (rtx, rrx) = mpsc::channel();
        let _ = tx.send(AppEvent::Approval {
            id,
            title: preview.title,
            body: preview.body,
            reply: rtx,
        });
        let Some(v) = wait_for(&rrx, cancel) else {
            log.log(
                "approval",
                json!({"decision": "cancelled", "headline": headline}),
            );
            return Gated::Cancelled;
        };
        let d = v.decision;
        gate.record(d);
        let note = v
            .note
            .as_deref()
            .map(str::trim)
            .filter(|n| !n.is_empty())
            .map(str::to_string);
        log.log(
            "approval",
            json!({"decision": d.label(), "headline": headline, "note": note}),
        );
        if !d.allows() {
            return Gated::Declined(note);
        }
    } else if needs_approval {
        log.log(
            "approval",
            json!({"decision": "auto-approved", "headline": headline}),
        );
    }

    let out = tools::execute(ws, call, cancel);
    log.log(
        "tool_result",
        json!({"name": call.name, "ok": out.ok, "summary": out.summary,
               "output": out.output.chars().take(4000).collect::<String>()}),
    );
    let _ = tx.send(AppEvent::Tool {
        id,
        headline,
        summary: out.summary.clone(),
        ok: out.ok,
    });
    Gated::Ran(out)
}

enum End {
    Done,
    Cancelled,
    OutOfSteps,
    Failed(String),
}

pub fn run(spec: RunSpec, id: u64, tx: Sender<AppEvent>, cancel: Arc<AtomicBool>) {
    let t = &spec.template;
    let ws = Workspace::new(&spec.root);
    let schema = tools::schema(&t.tools);
    let mut gate = Gate::new(spec.auto);
    let mut log = RunLog::create(&spec.log_dir, &t.name);
    log.log(
        "run_start",
        json!({
            "template": t.name, "model": spec.model, "task": spec.task,
            "workspace": ws.root.display().to_string(), "auto_approve": spec.auto,
            "tools": t.tools,
        }),
    );

    let mut messages = spec.history;
    if messages.is_empty() {
        messages.push(json!({"role": "system", "content": system_prompt(t, &ws.root)}));
    }
    messages.push(json!({"role": "user", "content": spec.task}));

    let mut steps = 0;
    let mut answer = String::new();
    // Exact calls the user already declined. Small models re-propose the
    // same write after a plain "no" (seen live with qwen3:8b); asking again
    // would loop until the step limit.
    let mut declined: Vec<String> = Vec::new();
    // Files read this run and not changed since. qwen3:8b re-reads a folder
    // of short files in a cycle instead of writing (seen in the task eval's
    // draft-hr-replies case: 0 of 3 runs wrote a draft). Sending the file
    // again feeds the loop; telling it the contents are already above breaks it.
    let mut read: HashSet<String> = HashSet::new();
    let end = 'run: loop {
        if steps >= t.max_steps {
            break End::OutOfSteps;
        }
        steps += 1;

        let started = Instant::now();
        let body = chat_body(
            &spec.url,
            &spec.model,
            messages.clone(),
            Some(schema.clone()),
        );
        let reply = ollama::chat(&spec.url, body, &cancel, |d| {
            if let Delta::Content(c) = d {
                let _ = tx.send(AppEvent::Token {
                    id,
                    delta: c.to_string(),
                });
            }
        });
        let reply = match reply {
            Ok(r) => r,
            Err(ChatError::Cancelled) => break End::Cancelled,
            Err(e) => break End::Failed(e.to_string()),
        };

        let content = render::strip_think(&reply.content);
        let mut calls = tools::from_native(&reply.tool_calls);
        let in_text = calls.is_empty();
        if in_text {
            calls = tools::from_content(&content, &t.tools);
        }
        let in_text = in_text && !calls.is_empty();
        log.log(
            "model_turn",
            json!({
                "step": steps, "duration_ms": started.elapsed().as_millis() as u64,
                "content": content, "tool_calls": native(&calls), "calls_parsed_from_text": in_text,
                "truncated": reply.truncated,
            }),
        );

        if reply.truncated {
            let _ = tx.send(AppEvent::Note {
                id,
                text: format!(
                    "the model's reply hit the {}-token cap (AMADEUS_NUM_PREDICT) and was cut off",
                    num_predict()
                ),
            });
        }

        if calls.is_empty() {
            let _ = tx.send(AppEvent::TurnEnd { id, keep: true });
            messages.push(json!({"role": "assistant", "content": content}));
            answer = content;
            break End::Done;
        }

        // Calls written as JSON text aren't prose; drop them from the screen.
        let keep = !in_text && !content.trim().is_empty();
        let _ = tx.send(AppEvent::TurnEnd { id, keep });
        messages.push(json!({
            "role": "assistant",
            "content": if in_text { String::new() } else { content },
            "tool_calls": native(&calls),
        }));

        for call in calls {
            let headline = call.headline();
            log.log(
                "tool_call",
                json!({"name": call.name, "args": call.args, "headline": headline}),
            );
            let result =
                |output: String| json!({"role": "tool", "tool_name": call.name, "content": output});

            let allowed = t.tools.contains(&call.name);
            let Some(ts) = tools::spec(&call.name).filter(|_| allowed) else {
                let msg = format!("{} isn't available here", call.name);
                let _ = tx.send(AppEvent::Tool {
                    id,
                    headline,
                    summary: msg.clone(),
                    ok: false,
                });
                log.log(
                    "tool_result",
                    json!({"name": call.name, "ok": false, "summary": msg}),
                );
                messages.push(result(format!(
                    "error: no tool named {}. Available: {}",
                    call.name,
                    t.tools.join(", ")
                )));
                continue;
            };

            let fingerprint = format!("{}{}", call.name, call.args);
            if ts.needs_approval && declined.contains(&fingerprint) {
                let msg = "already declined, not asked again".to_string();
                let _ = tx.send(AppEvent::Tool {
                    id,
                    headline,
                    summary: msg.clone(),
                    ok: false,
                });
                log.log(
                    "tool_result",
                    json!({"name": call.name, "ok": false, "summary": msg}),
                );
                messages.push(result(
                    "You proposed this exact call before and the user declined it. Do not try it \
                     again. Stop and reply with what you would change, or ask the user what they want."
                        .into(),
                ));
                continue;
            }

            let path = read_key(&call);
            if call.name == "read_file" && path.as_ref().is_some_and(|p| read.contains(p)) {
                let msg = "already read, not sent again".to_string();
                let _ = tx.send(AppEvent::Tool {
                    id,
                    headline,
                    summary: msg.clone(),
                    ok: true,
                });
                log.log(
                    "tool_result",
                    json!({"name": call.name, "ok": true, "summary": msg}),
                );
                messages.push(result(
                    "You already read this file in this run and it hasn't changed. Its contents are \
                     above. Don't read it again: use what you have and go on to the next step of \
                     your task."
                        .into(),
                ));
                continue;
            }

            let out = match approve_and_run(
                &ws,
                &mut gate,
                &call,
                ts.needs_approval,
                &mut log,
                &tx,
                id,
                &cancel,
            ) {
                Gated::Ran(out) => out,
                Gated::Cancelled => break 'run End::Cancelled,
                Gated::Refused(e) => {
                    messages.push(result(format!("error: {e}")));
                    continue;
                }
                Gated::Declined(note) => {
                    declined.push(fingerprint);
                    let (summary, told) = match &note {
                        Some(n) => (
                            format!("declined: {n}"),
                            format!(
                                "The user declined this {} call and said: \"{n}\". It was not run. \
                                 Fix that and propose the corrected call.",
                                call.name
                            ),
                        ),
                        None => (
                            "declined, not run".to_string(),
                            format!(
                                "The user declined this {} call. It was not run. Do not repeat it.",
                                call.name
                            ),
                        ),
                    };
                    let _ = tx.send(AppEvent::Tool {
                        id,
                        headline,
                        summary,
                        ok: false,
                    });
                    messages.push(result(told));
                    continue;
                }
            };
            match call.name.as_str() {
                "read_file" if out.ok => read.extend(path),
                "write_file" => {
                    if let Some(p) = &path {
                        read.remove(p);
                    }
                }
                // A command can change any file.
                "run_command" => read.clear(),
                _ => {}
            }
            messages.push(result(out.output));
            if cancel.load(Ordering::Relaxed) {
                break 'run End::Cancelled;
            }
        }
    };

    let status = match &end {
        End::Done => "done",
        End::Cancelled => "cancelled",
        End::OutOfSteps => "stopped: step limit",
        End::Failed(_) => "failed",
    };
    log.log(
        "run_end",
        json!({"status": status, "steps": steps, "answer": answer}),
    );
    let _ = tx.send(AppEvent::RunEnd {
        id,
        messages,
        log: log.path.clone(),
    });
    let _ = tx.send(match end {
        End::Done => AppEvent::Done { id },
        End::Cancelled => {
            let _ = tx.send(AppEvent::Note { id, text: "[cancelled]".into() });
            AppEvent::Done { id }
        }
        End::OutOfSteps => AppEvent::Failed {
            id,
            text: format!(
                "stopped after {} steps without a final answer. Try a narrower request, or the heavy tier (ctrl+t).",
                t.max_steps
            ),
        },
        End::Failed(e) => AppEvent::Failed { id, text: e },
    });
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::approval::Decision;
    use crate::templates;
    use std::collections::VecDeque;
    use std::fs;
    use std::io::Write;
    use std::net::TcpListener;
    use std::sync::Mutex;
    use std::thread;

    /// A fake Ollama: answers /api/show, and serves scripted /api/chat replies
    /// in order. Returns the base URL and every chat request body it saw.
    pub(crate) fn stub(replies: Vec<Value>) -> (String, Arc<Mutex<Vec<Value>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let queue = Arc::new(Mutex::new(VecDeque::from(replies)));
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen2 = seen.clone();
        thread::spawn(move || {
            for sock in listener.incoming() {
                let Ok(mut sock) = sock else { return };
                let (head, req) = crate::ollama::read_request(&mut sock);
                let body = if head.starts_with("POST /api/show") {
                    "{\"capabilities\":[\"completion\",\"tools\"]}".to_string()
                } else {
                    if let Ok(v) = serde_json::from_slice(&req) {
                        seen2.lock().unwrap().push(v);
                    }
                    let next = queue.lock().unwrap().pop_front();
                    let msg = next.unwrap_or(json!({"content": "out of script"}));
                    format!("{}\n", json!({"message": msg, "done": true}))
                };
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = sock.write_all(resp.as_bytes());
            }
        });
        (url, seen)
    }

    fn call(name: &str, args: Value) -> Value {
        json!({"content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]})
    }

    fn template(tools: &[&str]) -> Template {
        templates::parse(&format!(
            "name = \"t\"\ntitle = \"T\"\ndescription = \"d\"\nsystem = \"Be helpful.\"\nexample = \"x\"\ntools = [{}]\n",
            tools.iter().map(|t| format!("\"{t}\"")).collect::<Vec<_>>().join(", ")
        ))
        .unwrap()
    }

    struct Ran {
        events: Vec<String>,
        messages: Vec<Value>,
        log: String,
        ws: PathBuf,
        approvals: usize,
    }

    /// Run the loop against `replies`, answering every approval with `answer`.
    fn run_with(
        tag: &str,
        tools: &[&str],
        replies: Vec<Value>,
        answer: Decision,
        auto: bool,
    ) -> Ran {
        run_with_note(tag, tools, replies, Verdict::of(answer), auto)
    }

    fn run_with_note(
        tag: &str,
        tools: &[&str],
        replies: Vec<Value>,
        answer: Verdict,
        auto: bool,
    ) -> Ran {
        let ws = std::env::temp_dir().join(format!("amadeus-agent-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&ws);
        fs::create_dir_all(&ws).unwrap();
        fs::write(ws.join("a.txt"), "hello from a\n").unwrap();
        let (url, _) = stub(replies);
        let (tx, rx) = mpsc::channel();
        let spec = RunSpec {
            url,
            model: "m".into(),
            template: template(tools),
            root: ws.clone(),
            history: Vec::new(),
            task: "do the thing".into(),
            auto,
            log_dir: ws.join(".logs"),
        };
        let cancel = Arc::new(AtomicBool::new(false));
        thread::spawn(move || run(spec, 1, tx, cancel));

        let (mut events, mut messages, mut log, mut approvals) =
            (Vec::new(), Vec::new(), String::new(), 0);
        while let Ok(ev) = rx.recv_timeout(Duration::from_secs(10)) {
            match ev {
                AppEvent::Approval { reply, title, .. } => {
                    approvals += 1;
                    events.push(format!("approval: {title}"));
                    reply.send(answer.clone()).unwrap();
                }
                AppEvent::Tool {
                    headline,
                    summary,
                    ok,
                    ..
                } => events.push(format!("tool {headline} ok={ok} {summary}")),
                AppEvent::TurnEnd { keep, .. } => events.push(format!("turn keep={keep}")),
                AppEvent::RunEnd {
                    messages: m,
                    log: l,
                    ..
                } => {
                    messages = m;
                    log = fs::read_to_string(l).unwrap_or_default();
                }
                AppEvent::Done { .. } => {
                    events.push("done".into());
                    break;
                }
                AppEvent::Failed { text, .. } => {
                    events.push(format!("failed: {text}"));
                    break;
                }
                _ => {}
            }
        }
        Ran {
            events,
            messages,
            log,
            ws,
            approvals,
        }
    }

    #[test]
    fn approved_write_lands_and_everything_is_logged() {
        let r = run_with(
            "yes",
            &["list_dir", "read_file", "write_file"],
            vec![
                call("list_dir", json!({"path": "."})),
                call(
                    "write_file",
                    json!({"path": "out/summary.md", "content": "# Summary\n"}),
                ),
                json!({"content": "Saved out/summary.md"}),
            ],
            Decision::Approve,
            false,
        );
        assert_eq!(r.approvals, 1);
        assert!(
            r.events
                .contains(&"approval: create out/summary.md (1 lines)".to_string()),
            "{:?}",
            r.events
        );
        assert_eq!(
            fs::read_to_string(r.ws.join("out/summary.md")).unwrap(),
            "# Summary\n"
        );
        assert_eq!(r.events.last().unwrap(), "done");

        let roles: Vec<&str> = r
            .messages
            .iter()
            .map(|m| m["role"].as_str().unwrap())
            .collect();
        assert_eq!(
            roles,
            [
                "system",
                "user",
                "assistant",
                "tool",
                "assistant",
                "tool",
                "assistant"
            ]
        );
        assert!(r.messages[3]["content"].as_str().unwrap().contains("a.txt"));

        for kind in [
            "run_start",
            "model_turn",
            "tool_call",
            "approval",
            "tool_result",
            "run_end",
        ] {
            assert!(
                r.log.contains(&format!("\"kind\":\"{kind}\"")),
                "log is missing {kind}:\n{}",
                r.log
            );
        }
        assert!(r.log.contains("\"decision\":\"approved\""));
        let _ = fs::remove_dir_all(r.ws);
    }

    #[test]
    fn declined_write_is_not_run_and_the_model_is_told() {
        let r = run_with(
            "no",
            &["write_file"],
            vec![
                call("write_file", json!({"path": "x.md", "content": "x"})),
                json!({"content": "OK, I won't write it."}),
            ],
            Decision::Deny,
            false,
        );
        assert!(!r.ws.join("x.md").exists());
        assert!(
            r.events.iter().any(|e| e.contains("declined, not run")),
            "{:?}",
            r.events
        );
        let tool_msg = r.messages.iter().find(|m| m["role"] == "tool").unwrap();
        assert!(tool_msg["content"].as_str().unwrap().contains("declined"));
        assert!(r.log.contains("\"decision\":\"declined\""));
        let _ = fs::remove_dir_all(r.ws);
    }

    #[test]
    fn a_repeated_declined_call_is_refused_without_asking() {
        let write = call("write_file", json!({"path": "x.md", "content": "x"}));
        let r = run_with(
            "repeat",
            &["write_file"],
            vec![
                write.clone(),
                write,
                json!({"content": "What should I change?"}),
            ],
            Decision::Deny,
            false,
        );
        assert_eq!(r.approvals, 1, "{:?}", r.events);
        assert!(
            r.events.iter().any(|e| e.contains("already declined")),
            "{:?}",
            r.events
        );
        assert_eq!(r.events.last().unwrap(), "done");
        let _ = fs::remove_dir_all(r.ws);
    }

    #[test]
    fn a_repeated_read_is_not_sent_again() {
        let r = run_with(
            "reread",
            &["read_file"],
            vec![
                call("read_file", json!({"path": "a.txt"})),
                call("read_file", json!({"path": "./a.txt"})),
                json!({"content": "done"}),
            ],
            Decision::Approve,
            false,
        );
        let tools: Vec<&str> = r
            .messages
            .iter()
            .filter(|m| m["role"] == "tool")
            .map(|m| m["content"].as_str().unwrap())
            .collect();
        assert_eq!(tools.len(), 2, "{tools:?}");
        assert!(tools[0].contains("hello from a"), "{tools:?}");
        assert!(tools[1].contains("already read"), "{tools:?}");
        assert!(r.log.contains("already read, not sent again"));
        assert_eq!(r.events.last().unwrap(), "done");
        let _ = fs::remove_dir_all(r.ws);
    }

    #[test]
    fn a_read_after_a_write_to_that_file_gets_the_new_contents() {
        let r = run_with(
            "reread-write",
            &["read_file", "write_file"],
            vec![
                call("read_file", json!({"path": "a.txt"})),
                call(
                    "write_file",
                    json!({"path": "a.txt", "content": "changed\n"}),
                ),
                call("read_file", json!({"path": "a.txt"})),
                json!({"content": "done"}),
            ],
            Decision::Approve,
            true,
        );
        let last = r
            .messages
            .iter()
            .rev()
            .find(|m| m["role"] == "tool")
            .unwrap();
        assert!(
            last["content"].as_str().unwrap().contains("changed"),
            "{last}"
        );
        let _ = fs::remove_dir_all(r.ws);
    }

    #[test]
    fn a_decline_note_reaches_the_model() {
        let r = run_with_note(
            "note",
            &["write_file"],
            vec![
                call("write_file", json!({"path": "x.csv", "content": "a,Theo"})),
                json!({"content": "ok"}),
            ],
            Verdict {
                decision: Decision::Deny,
                note: Some("leave paid_by empty".into()),
            },
            false,
        );
        let tool_msg = r.messages.iter().find(|m| m["role"] == "tool").unwrap();
        assert!(tool_msg["content"]
            .as_str()
            .unwrap()
            .contains("leave paid_by empty"));
        assert!(
            r.events
                .iter()
                .any(|e| e.contains("declined: leave paid_by empty")),
            "{:?}",
            r.events
        );
        assert!(r.log.contains("\"note\":\"leave paid_by empty\""));
        let _ = fs::remove_dir_all(r.ws);
    }

    #[test]
    fn approve_rest_stops_asking_for_the_rest_of_the_run() {
        let r = run_with(
            "rest",
            &["write_file"],
            vec![
                call("write_file", json!({"path": "1.md", "content": "1"})),
                call("write_file", json!({"path": "2.md", "content": "2"})),
                json!({"content": "both written"}),
            ],
            Decision::ApproveRest,
            false,
        );
        assert_eq!(r.approvals, 1);
        assert!(r.ws.join("1.md").exists() && r.ws.join("2.md").exists());
        assert!(r.log.contains("auto-approved"));
        let _ = fs::remove_dir_all(r.ws);
    }

    #[test]
    fn auto_approve_never_asks() {
        let r = run_with(
            "auto",
            &["write_file"],
            vec![
                call("write_file", json!({"path": "1.md", "content": "1"})),
                json!({"content": "done"}),
            ],
            Decision::Deny,
            true,
        );
        assert_eq!(r.approvals, 0);
        assert!(r.ws.join("1.md").exists());
        let _ = fs::remove_dir_all(r.ws);
    }

    #[test]
    fn calls_written_as_text_run_and_their_json_is_hidden() {
        let r = run_with(
            "text",
            &["read_file"],
            vec![
                json!({"content": "{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.txt\"}}"}),
                json!({"content": "It says hello."}),
            ],
            Decision::Approve,
            false,
        );
        assert_eq!(r.events[0], "turn keep=false");
        assert!(
            r.events
                .iter()
                .any(|e| e.starts_with("tool read_file a.txt ok=true")),
            "{:?}",
            r.events
        );
        let tool_msg = r.messages.iter().find(|m| m["role"] == "tool").unwrap();
        assert_eq!(tool_msg["content"], "hello from a\n");
        let _ = fs::remove_dir_all(r.ws);
    }

    #[test]
    fn tools_outside_the_template_are_refused_without_asking() {
        let r = run_with(
            "deny",
            &["read_file"],
            vec![
                call("run_command", json!({"command": "touch pwned"})),
                json!({"content": "fine"}),
            ],
            Decision::Approve,
            false,
        );
        assert_eq!(r.approvals, 0);
        assert!(!r.ws.join("pwned").exists());
        assert!(
            r.events.iter().any(|e| e.contains("isn't available here")),
            "{:?}",
            r.events
        );
        let _ = fs::remove_dir_all(r.ws);
    }

    #[test]
    fn step_limit_ends_the_run_with_a_message() {
        let mut t = template(&["read_file"]);
        t.max_steps = 2;
        let loops = vec![call("read_file", json!({"path": "a.txt"})); 5];
        let ws = std::env::temp_dir().join(format!("amadeus-agent-steps-{}", std::process::id()));
        fs::create_dir_all(&ws).unwrap();
        fs::write(ws.join("a.txt"), "x").unwrap();
        let (url, seen) = stub(loops);
        let (tx, rx) = mpsc::channel();
        let spec = RunSpec {
            url,
            model: "m".into(),
            template: t,
            root: ws.clone(),
            history: Vec::new(),
            task: "loop".into(),
            auto: false,
            log_dir: ws.join(".logs"),
        };
        thread::spawn(move || run(spec, 1, tx, Arc::new(AtomicBool::new(false))));
        let failed = loop {
            match rx.recv_timeout(Duration::from_secs(10)).unwrap() {
                AppEvent::Failed { text, .. } => break text,
                AppEvent::Done { .. } => panic!("should not finish"),
                _ => {}
            }
        };
        assert!(failed.contains("stopped after 2 steps"), "{failed}");
        assert_eq!(seen.lock().unwrap().len(), 2);
        let _ = fs::remove_dir_all(ws);
    }
}
