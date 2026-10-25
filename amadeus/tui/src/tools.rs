//! The tool registry: what the model may call, which calls need approval,
//! and the workspace confinement every path goes through.

use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use serde_json::{json, Value};

use crate::proc;
use crate::rag::{self, QueryOutcome};

/// Largest file `read_file` hands to the model, in characters. Small local
/// models have 8k-token contexts; one big file shouldn't evict everything.
const READ_CAP: usize = 12_000;
const OUTPUT_CAP: usize = 4_000;
const LIST_CAP: usize = 300;
const SEARCH_CAP: usize = 80;

/// Directories no tool descends into.
const SKIP_DIRS: &[&str] = &[
    ".git",
    ".amadeus",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "target",
    "dist",
    "build",
    ".cache",
];

pub struct ToolSpec {
    pub name: &'static str,
    pub description: &'static str,
    pub needs_approval: bool,
    params: fn() -> Value,
}

fn obj(props: Value, required: &[&str]) -> Value {
    json!({ "type": "object", "properties": props, "required": required })
}

pub const TOOLS: &[ToolSpec] = &[
    ToolSpec {
        name: "list_dir",
        description: "List files and folders in a workspace directory. Use '.' for the workspace root. Set recursive to true to include subfolders.",
        needs_approval: false,
        params: || {
            obj(
                json!({
                    "path": {"type": "string", "description": "directory relative to the workspace, '.' for the root"},
                    "recursive": {"type": "boolean", "description": "include subfolders (default false)"}
                }),
                &["path"],
            )
        },
    },
    ToolSpec {
        name: "read_file",
        description: "Read a text file from the workspace.",
        needs_approval: false,
        params: || {
            obj(
                json!({"path": {"type": "string", "description": "file path relative to the workspace"}}),
                &["path"],
            )
        },
    },
    ToolSpec {
        name: "search_files",
        description: "Find lines containing some text (case-insensitive) in workspace files. Returns path:line: text.",
        needs_approval: false,
        params: || {
            obj(
                json!({
                    "query": {"type": "string", "description": "text to look for"},
                    "path": {"type": "string", "description": "folder to search, default '.'"}
                }),
                &["query"],
            )
        },
    },
    ToolSpec {
        name: "rag_query",
        description: "Semantic search over the indexed codebase. Returns the most relevant code passages with file paths and line ranges.",
        needs_approval: false,
        params: || {
            obj(
                json!({"question": {"type": "string", "description": "a natural-language question about the code"}}),
                &["question"],
            )
        },
    },
    ToolSpec {
        name: "write_file",
        description: "Create or overwrite a text file in the workspace with the complete content. The user approves every write.",
        needs_approval: true,
        params: || {
            obj(
                json!({
                    "path": {"type": "string", "description": "file path relative to the workspace"},
                    "content": {"type": "string", "description": "the full file content"}
                }),
                &["path", "content"],
            )
        },
    },
    ToolSpec {
        name: "run_command",
        description: "Run a shell command in the workspace directory and return its output. The user approves every command.",
        needs_approval: true,
        params: || {
            obj(
                json!({"command": {"type": "string", "description": "the shell command"}}),
                &["command"],
            )
        },
    },
];

pub fn spec(name: &str) -> Option<&'static ToolSpec> {
    TOOLS.iter().find(|t| t.name == name)
}

/// The `tools` field for /api/chat, restricted to `allowed`.
pub fn schema(allowed: &[String]) -> Value {
    Value::Array(
        TOOLS
            .iter()
            .filter(|t| allowed.iter().any(|a| a == t.name))
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": (t.params)(),
                    }
                })
            })
            .collect(),
    )
}

#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    pub name: String,
    pub args: Value,
}

impl Call {
    fn str_arg(&self, k: &str) -> Result<&str, String> {
        self.args
            .get(k)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("missing string argument `{k}`"))
    }

    /// One line for the scrollback: `read_file notes/a.txt`.
    pub fn headline(&self) -> String {
        let main = match self.name.as_str() {
            "list_dir" | "read_file" | "write_file" => self.args["path"].as_str(),
            "search_files" => self.args["query"].as_str(),
            "rag_query" => self.args["question"].as_str(),
            "run_command" => self.args["command"].as_str(),
            _ => None,
        };
        match main {
            Some(m) => format!("{} {m}", self.name),
            None => format!("{} {}", self.name, self.args),
        }
    }
}

/// Calls from Ollama's `tool_calls`. Arguments sometimes arrive as a JSON
/// string rather than an object; both are accepted.
pub fn from_native(calls: &[Value]) -> Vec<Call> {
    calls
        .iter()
        .filter_map(|c| {
            let f = c.get("function").unwrap_or(c);
            let name = f.get("name")?.as_str()?.to_string();
            let args = match f.get("arguments") {
                Some(Value::String(s)) => serde_json::from_str(s).unwrap_or(json!({})),
                Some(v) => v.clone(),
                None => json!({}),
            };
            Some(Call { name, args })
        })
        .collect()
}

/// Calls written into `content` instead of `tool_calls`: what qwen2.5-coder
/// does (measured, see docs/REVAMP.md). Accepts bare objects, objects in
/// ``` fences or `<tool_call>` tags, several per reply. Only names in
/// `allowed` count, so ordinary JSON in an answer isn't executed.
pub fn from_content(content: &str, allowed: &[String]) -> Vec<Call> {
    let mut out = Vec::new();
    let b = content.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] != b'{' {
            i += 1;
            continue;
        }
        let Some(end) = balanced_end(content, i) else {
            break;
        };
        if let Ok(v) = serde_json::from_str::<Value>(&content[i..=end]) {
            if let Some(c) = from_native(std::slice::from_ref(&v)).pop() {
                if allowed.contains(&c.name) {
                    out.push(c);
                    i = end + 1;
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

/// Index of the `}` closing the object that opens at `start`, aware of
/// strings and escapes.
fn balanced_end(s: &str, start: usize) -> Option<usize> {
    let (mut depth, mut in_str, mut esc) = (0i32, false, false);
    for (off, ch) in s[start..].char_indices() {
        if in_str {
            match (esc, ch) {
                (true, _) => esc = false,
                (false, '\\') => esc = true,
                (false, '"') => in_str = false,
                _ => {}
            }
            continue;
        }
        match ch {
            '"' => in_str = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(start + off);
                }
            }
            _ => {}
        }
    }
    None
}

// ---------- workspace ----------

pub struct Workspace {
    pub root: PathBuf,
}

impl Workspace {
    pub fn new(root: &Path) -> Workspace {
        Workspace {
            root: root.canonicalize().unwrap_or_else(|_| root.to_path_buf()),
        }
    }

    /// Resolve a model-supplied path and refuse anything outside the
    /// workspace: absolute paths elsewhere, `..`, and symlinks (including
    /// dangling ones) that point out. For a path that doesn't exist yet, the
    /// nearest existing ancestor is resolved and the rest appended.
    pub fn resolve(&self, rel: &str) -> Result<PathBuf, String> {
        let rel = rel.trim();
        let p = Path::new(if rel.is_empty() { "." } else { rel });
        let joined = if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.root.join(p)
        };
        let mut norm = PathBuf::new();
        for c in joined.components() {
            match c {
                Component::ParentDir => {
                    norm.pop();
                }
                Component::CurDir => {}
                other => norm.push(other),
            }
        }
        let mut existing = norm.clone();
        let mut rest = Vec::new();
        while existing.symlink_metadata().is_err() {
            match existing.file_name() {
                Some(n) => rest.push(n.to_os_string()),
                None => break,
            }
            if !existing.pop() {
                break;
            }
        }
        let mut real = existing
            .canonicalize()
            .map_err(|e| format!("can't resolve {rel}: {e}"))?;
        for r in rest.iter().rev() {
            real.push(r);
        }
        if real.starts_with(&self.root) {
            Ok(real)
        } else {
            Err(format!("refused: {rel} is outside the workspace"))
        }
    }

    pub fn display(&self, p: &Path) -> String {
        let r = p.strip_prefix(&self.root).unwrap_or(p);
        let s = r.display().to_string();
        if s.is_empty() {
            ".".into()
        } else {
            s
        }
    }
}

// ---------- approval preview ----------

/// What the user sees before approving a call.
pub struct Preview {
    pub title: String,
    pub body: String,
}

pub fn preview(ws: &Workspace, call: &Call) -> Result<Preview, String> {
    match call.name.as_str() {
        "write_file" => {
            let path = call.str_arg("path")?;
            let content = call.str_arg("content")?;
            let target = ws.resolve(path)?;
            let old = fs::read_to_string(&target).ok();
            let shown = ws.display(&target);
            let (title, body) = match &old {
                None => (
                    format!("create {shown} ({} lines)", content.lines().count()),
                    content
                        .lines()
                        .map(|l| format!("+{l}"))
                        .collect::<Vec<_>>()
                        .join("\n"),
                ),
                Some(o) if o == content => (format!("rewrite {shown} (no change)"), String::new()),
                Some(o) => (
                    format!("change {shown}"),
                    similar::TextDiff::from_lines(o.as_str(), content)
                        .unified_diff()
                        .context_radius(2)
                        .to_string(),
                ),
            };
            Ok(Preview { title, body })
        }
        "run_command" => Ok(Preview {
            title: "run a command in the workspace".into(),
            body: format!("$ {}", call.str_arg("command")?),
        }),
        other => Ok(Preview {
            title: other.to_string(),
            body: call.args.to_string(),
        }),
    }
}

// ---------- execution ----------

pub struct Outcome {
    pub ok: bool,
    /// Sent back to the model.
    pub output: String,
    /// One short line for the scrollback.
    pub summary: String,
}

fn ok(output: String, summary: String) -> Outcome {
    Outcome {
        ok: true,
        output,
        summary,
    }
}

fn fail(msg: String) -> Outcome {
    Outcome {
        ok: false,
        output: format!("error: {msg}"),
        summary: msg,
    }
}

fn cap_tail(s: &str, n: usize) -> String {
    let count = s.chars().count();
    if count <= n {
        return s.to_string();
    }
    let tail: String = s.chars().skip(count - n).collect();
    format!("[… {} earlier characters cut]\n{tail}", count - n)
}

pub fn execute(ws: &Workspace, call: &Call, cancel: &AtomicBool) -> Outcome {
    let r = match call.name.as_str() {
        "list_dir" => list_dir(ws, call),
        "read_file" => read_file(ws, call),
        "search_files" => search_files(ws, call),
        "write_file" => write_file(ws, call),
        "run_command" => run_command(ws, call, cancel),
        "rag_query" => rag_query(ws, call, cancel),
        other => Err(format!("unknown tool {other}")),
    };
    r.unwrap_or_else(fail)
}

fn walk(dir: &Path, depth: usize, max_depth: usize, out: &mut Vec<(PathBuf, bool, u64)>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    let mut items: Vec<_> = rd.flatten().collect();
    items.sort_by_key(|e| e.file_name());
    for e in items {
        if out.len() >= LIST_CAP {
            return;
        }
        let name = e.file_name().to_string_lossy().to_string();
        let Ok(md) = e.metadata() else { continue };
        if md.is_dir() {
            if SKIP_DIRS.contains(&name.as_str()) {
                continue;
            }
            out.push((e.path(), true, 0));
            if depth < max_depth {
                walk(&e.path(), depth + 1, max_depth, out);
            }
        } else {
            out.push((e.path(), false, md.len()));
        }
    }
}

fn human(n: u64) -> String {
    match n {
        n if n < 1024 => format!("{n} B"),
        n if n < 1024 * 1024 => format!("{:.1} KB", n as f64 / 1024.0),
        n => format!("{:.1} MB", n as f64 / 1024.0 / 1024.0),
    }
}

fn list_dir(ws: &Workspace, call: &Call) -> Result<Outcome, String> {
    let dir = ws.resolve(call.args["path"].as_str().unwrap_or("."))?;
    if !dir.is_dir() {
        return Err(format!("{} is not a folder", ws.display(&dir)));
    }
    let recursive = call.args["recursive"].as_bool().unwrap_or(false);
    let mut items = Vec::new();
    walk(&dir, 0, if recursive { 4 } else { 0 }, &mut items);
    let lines: Vec<String> = items
        .iter()
        .map(|(p, is_dir, size)| {
            let shown = ws.display(p);
            if *is_dir {
                format!("{shown}/")
            } else {
                format!("{shown}  ({})", human(*size))
            }
        })
        .collect();
    let files = items.iter().filter(|i| !i.1).count();
    let mut output = lines.join("\n");
    if items.len() >= LIST_CAP {
        output.push_str(&format!("\n[stopped at {LIST_CAP} entries]"));
    }
    if output.is_empty() {
        output = "(empty folder)".into();
    }
    Ok(ok(output, format!("{files} files in {}", ws.display(&dir))))
}

fn read_file(ws: &Workspace, call: &Call) -> Result<Outcome, String> {
    let path = ws.resolve(call.str_arg("path")?)?;
    let bytes = fs::read(&path).map_err(|e| format!("{}: {e}", ws.display(&path)))?;
    if bytes.iter().take(8192).any(|b| *b == 0) {
        return Err(format!("{} is a binary file", ws.display(&path)));
    }
    let text = String::from_utf8_lossy(&bytes);
    let lines = text.lines().count();
    let n = text.chars().count();
    let body = if n > READ_CAP {
        let head: String = text.chars().take(READ_CAP).collect();
        format!("{head}\n[… truncated: showing {READ_CAP} of {n} characters]")
    } else {
        text.to_string()
    };
    Ok(ok(body, format!("{lines} lines")))
}

fn search_files(ws: &Workspace, call: &Call) -> Result<Outcome, String> {
    let q = call.str_arg("query")?.to_lowercase();
    if q.trim().is_empty() {
        return Err("empty query".into());
    }
    let dir = ws.resolve(call.args["path"].as_str().unwrap_or("."))?;
    let mut items = Vec::new();
    walk(&dir, 0, 6, &mut items);
    let mut hits = Vec::new();
    'files: for (p, is_dir, size) in &items {
        if *is_dir || *size > 2_000_000 {
            continue;
        }
        let Ok(text) = fs::read_to_string(p) else {
            continue;
        };
        for (n, line) in text.lines().enumerate() {
            if line.to_lowercase().contains(&q) {
                let line: String = line.trim().chars().take(200).collect();
                hits.push(format!("{}:{}: {line}", ws.display(p), n + 1));
                if hits.len() >= SEARCH_CAP {
                    break 'files;
                }
            }
        }
    }
    let summary = format!("{} matches", hits.len());
    let output = if hits.is_empty() {
        "no matches".into()
    } else {
        hits.join("\n")
    };
    Ok(ok(output, summary))
}

fn write_file(ws: &Workspace, call: &Call) -> Result<Outcome, String> {
    let path = ws.resolve(call.str_arg("path")?)?;
    let content = call.str_arg("content")?;
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    fs::write(&path, content).map_err(|e| format!("{}: {e}", ws.display(&path)))?;
    let lines = content.lines().count();
    Ok(ok(
        format!("wrote {} ({lines} lines)", ws.display(&path)),
        format!("wrote {} lines", lines),
    ))
}

fn run_command(ws: &Workspace, call: &Call, cancel: &AtomicBool) -> Result<Outcome, String> {
    let cmd = call.str_arg("command")?;
    let mut c = Command::new("sh");
    c.arg("-c").arg(cmd).current_dir(&ws.root);
    let out = proc::run(c, cancel, Some(Duration::from_secs(120)), |_, _| {})
        .map_err(|e| format!("couldn't start: {e}"))?;
    let combined = format!("{}{}", out.stdout, out.stderr);
    let body = cap_tail(combined.trim_end(), OUTPUT_CAP);
    Ok(Outcome {
        ok: out.success(),
        output: format!("{}\n{body}", out.describe()),
        summary: out.describe(),
    })
}

fn rag_query(ws: &Workspace, call: &Call, cancel: &AtomicBool) -> Result<Outcome, String> {
    let q = call.str_arg("question")?;
    match rag::query(&ws.root, q, cancel) {
        QueryOutcome::Ok(r) => {
            // Hand the model the passages, not rag.py's answer instructions.
            let ctx = r
                .prompt
                .split_once("--- Context ---")
                .and_then(|(_, rest)| rest.split_once("--- End context ---"))
                .map(|(c, _)| c.trim().to_string())
                .unwrap_or(r.prompt);
            Ok(ok(
                cap_tail(&ctx, READ_CAP),
                format!("{} passages: {}", r.sources.len(), r.sources.join(", ")),
            ))
        }
        QueryOutcome::NoContext => Ok(ok(
            "no relevant passages in the index".into(),
            "nothing relevant".into(),
        )),
        QueryOutcome::Cancelled => Err("cancelled".into()),
        QueryOutcome::Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_ws(tag: &str) -> (PathBuf, Workspace) {
        let d = std::env::temp_dir().join(format!("amadeus-ws-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(d.join("notes")).unwrap();
        fs::write(d.join("notes/a.txt"), "alpha\nBeta line\n").unwrap();
        let ws = Workspace::new(&d);
        (d, ws)
    }

    fn call(name: &str, args: Value) -> Call {
        Call {
            name: name.into(),
            args,
        }
    }

    fn all() -> Vec<String> {
        TOOLS.iter().map(|t| t.name.to_string()).collect()
    }

    #[test]
    fn writes_and_commands_need_approval_reads_do_not() {
        for t in TOOLS {
            let expected = matches!(t.name, "write_file" | "run_command");
            assert_eq!(t.needs_approval, expected, "{}", t.name);
        }
    }

    #[test]
    fn confinement_refuses_escapes() {
        let (d, ws) = tmp_ws("conf");
        assert!(ws.resolve("../outside.txt").is_err());
        assert!(ws.resolve("notes/../../x").is_err());
        assert!(ws.resolve("/etc/passwd").is_err());
        assert!(
            ws.resolve("notes/new/deep.txt").is_ok(),
            "new paths inside are fine"
        );
        assert!(ws
            .resolve(&d.join("notes/a.txt").display().to_string())
            .is_ok());
        let _ = fs::remove_dir_all(d);
    }

    #[test]
    fn confinement_refuses_symlinks_out_even_dangling_ones() {
        let (d, ws) = tmp_ws("link");
        std::os::unix::fs::symlink("/etc", d.join("etc")).unwrap();
        std::os::unix::fs::symlink("/nonexistent/target", d.join("dangling")).unwrap();
        assert!(ws.resolve("etc/passwd").is_err());
        assert!(ws.resolve("dangling").is_err());
        let r = execute(
            &ws,
            &call("write_file", json!({"path": "etc/x", "content": "x"})),
            &AtomicBool::new(false),
        );
        assert!(!r.ok);
        let _ = fs::remove_dir_all(d);
    }

    #[test]
    fn every_tool_is_confined() {
        let (d, ws) = tmp_ws("every");
        let cancel = AtomicBool::new(false);
        for (name, args) in [
            ("list_dir", json!({"path": ".."})),
            ("read_file", json!({"path": "../../etc/passwd"})),
            ("search_files", json!({"query": "root", "path": "/etc"})),
            (
                "write_file",
                json!({"path": "/tmp/escape.txt", "content": "x"}),
            ),
        ] {
            let r = execute(&ws, &call(name, args), &cancel);
            assert!(!r.ok, "{name} escaped");
            assert!(
                r.output.contains("outside the workspace"),
                "{name}: {}",
                r.output
            );
        }
        let _ = fs::remove_dir_all(d);
    }

    #[test]
    fn read_list_search_write_round_trip() {
        let (d, ws) = tmp_ws("rt");
        let cancel = AtomicBool::new(false);
        let r = execute(
            &ws,
            &call("list_dir", json!({"path": ".", "recursive": true})),
            &cancel,
        );
        assert!(r.output.contains("notes/a.txt"), "{}", r.output);
        let r = execute(
            &ws,
            &call("search_files", json!({"query": "beta"})),
            &cancel,
        );
        assert_eq!(r.output, "notes/a.txt:2: Beta line");
        let r = execute(
            &ws,
            &call(
                "write_file",
                json!({"path": "out/s.csv", "content": "a,b\n1,2\n"}),
            ),
            &cancel,
        );
        assert!(r.ok, "{}", r.output);
        let r = execute(
            &ws,
            &call("read_file", json!({"path": "out/s.csv"})),
            &cancel,
        );
        assert_eq!(r.output, "a,b\n1,2\n");
        let _ = fs::remove_dir_all(d);
    }

    #[test]
    fn preview_shows_a_diff_against_disk() {
        let (d, ws) = tmp_ws("diff");
        let p = preview(
            &ws,
            &call(
                "write_file",
                json!({"path": "notes/a.txt", "content": "alpha\nGamma line\n"}),
            ),
        )
        .unwrap();
        assert_eq!(p.title, "change notes/a.txt");
        assert!(p.body.contains("-Beta line"), "{}", p.body);
        assert!(p.body.contains("+Gamma line"), "{}", p.body);
        let p = preview(
            &ws,
            &call("write_file", json!({"path": "new.md", "content": "# Hi\n"})),
        )
        .unwrap();
        assert_eq!(p.title, "create new.md (1 lines)");
        assert_eq!(p.body, "+# Hi");
        let p = preview(&ws, &call("run_command", json!({"command": "ls -la"}))).unwrap();
        assert_eq!(p.body, "$ ls -la");
        let _ = fs::remove_dir_all(d);
    }

    #[test]
    fn native_calls_accept_string_arguments() {
        let calls = from_native(&[
            json!({"function": {"name": "read_file", "arguments": "{\"path\":\"a\"}"}}),
        ]);
        assert_eq!(calls, vec![call("read_file", json!({"path": "a"}))]);
    }

    #[test]
    fn content_calls_are_parsed_like_qwen_coder_writes_them() {
        // Verbatim shape from the qwen2.5-coder:3b probe.
        let c = "{\"name\": \"list_dir\", \"arguments\": {\"path\": \"./notes/\"}}\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"./notes/notes.txt\"}}";
        let calls = from_content(c, &all());
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].args["path"], "./notes/notes.txt");

        let fenced = "Sure:\n```json\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"x\", \"content\": \"a } b\"}}\n```";
        let calls = from_content(fenced, &all());
        assert_eq!(
            calls[0].args["content"], "a } b",
            "braces in strings don't end the object"
        );
    }

    #[test]
    fn content_json_that_is_not_an_allowed_call_is_ignored() {
        assert!(from_content("{\"name\": \"Alice\", \"age\": 3}", &all()).is_empty());
        let only_read = vec!["read_file".to_string()];
        assert!(from_content(
            "{\"name\": \"run_command\", \"arguments\": {\"command\": \"rm -rf /\"}}",
            &only_read
        )
        .is_empty());
    }

    #[test]
    fn command_output_is_capped_from_the_front() {
        let s = "x".repeat(10) + "END";
        let c = cap_tail(&s, 5);
        assert!(c.ends_with("xxEND"));
        assert!(c.starts_with("[… 8 earlier"));
    }
}
