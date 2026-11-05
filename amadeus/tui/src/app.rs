use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ratatui::crossterm::event::{
    Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEventKind,
};
use serde_json::{json, Value};

use crate::agent::{self, RunSpec};
use crate::approval::{Decision, Verdict};
use crate::event::AppEvent;
use crate::health::{self, Health, Status};
use crate::hint;
use crate::history::InputHistory;
use crate::ollama::{self, ChatError, Delta};
use crate::paths;
use crate::proc;
use crate::rag::{self, QueryOutcome};
use crate::render;
use crate::scrollback::{Kind, Scrollback, View};
use crate::templates::Template;

const CHAT_PROMPT: &str = "You are AMADEUS, a helpful assistant running entirely on the \
user's own computer. Answer clearly and concisely. Use markdown when it helps.";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tier {
    Fast,
    Main,
    Heavy,
}

impl Tier {
    pub fn label(self) -> &'static str {
        match self {
            Tier::Fast => "fast",
            Tier::Main => "main",
            Tier::Heavy => "heavy",
        }
    }

    pub fn next(self) -> Tier {
        match self {
            Tier::Fast => Tier::Main,
            Tier::Main => Tier::Heavy,
            Tier::Heavy => Tier::Fast,
        }
    }
}

pub struct Config {
    /// The workspace AMADEUS operates on: the directory it was launched in.
    pub root: PathBuf,
    pub ollama_url: String,
    pub fast_model: String,
    pub main_model: String,
    pub heavy_model: String,
    pub history_file: PathBuf,
    pub runs_dir: PathBuf,
    pub session: u64,
    pub shell_timeout: Duration,
}

impl Config {
    pub fn from_env() -> Config {
        let root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let root = root.canonicalize().unwrap_or(root);
        Config::for_root(root)
    }

    pub fn for_root(root: PathBuf) -> Config {
        Config {
            history_file: paths::history_file(&root),
            runs_dir: paths::runs_dir(&root),
            root,
            ollama_url: env_or("AMADEUS_OLLAMA_URL", "http://localhost:11434"),
            fast_model: env_or("AMADEUS_FAST_MODEL", "qwen3:4b-instruct"),
            main_model: env_or("AMADEUS_MAIN_MODEL", "qwen3:8b"),
            heavy_model: env_or("AMADEUS_HEAVY_MODEL", "qwen3:14b"),
            session: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            shell_timeout: Duration::from_secs(
                env_or("AMADEUS_SHELL_TIMEOUT", "120")
                    .parse()
                    .unwrap_or(120),
            ),
        }
    }

    pub fn model(&self, tier: Tier) -> &str {
        match tier {
            Tier::Fast => &self.fast_model,
            Tier::Main => &self.main_model,
            Tier::Heavy => &self.heavy_model,
        }
    }

    pub fn tiers(&self) -> Vec<(&'static str, String)> {
        [Tier::Fast, Tier::Main, Tier::Heavy]
            .iter()
            .map(|t| (t.label(), self.model(*t).to_string()))
            .collect()
    }
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.to_string())
}

/// Request body for one chat turn. `think: false` goes only to models that
/// advertise thinking; any `<think>` that still leaks into content is folded
/// by the renderer.
///
/// Every request carries the same `num_ctx`. Ollama reloads a model whenever
/// it changes, and the default differs by version (16k on 0.40), so a chat
/// followed by a task would otherwise reload, and spill to CPU, each time.
/// `num_predict` caps one turn so a runaway generation can't hold the GPU.
pub fn chat_body(url: &str, model: &str, messages: Vec<Value>, tools: Option<Value>) -> Value {
    let mut body = json!({
        "model": model,
        "messages": messages,
        "options": {
            "temperature": 0.2,
            "num_ctx": agent::num_ctx(),
            "num_predict": agent::num_predict(),
        },
    });
    let thinks = ollama::capabilities(url, model)
        .map(|c| c.iter().any(|x| x == "thinking"))
        .unwrap_or(false);
    if thinks {
        body["think"] = json!(false);
    }
    if let Some(t) = tools {
        body["tools"] = t;
    }
    body
}

/// A side-effecting tool call waiting for y / n / a.
pub struct Pending {
    pub title: String,
    pub body: String,
    pub scroll: usize,
    /// Some while the user types why they're declining (after `n`).
    pub note: Option<String>,
    reply: Sender<Verdict>,
}

/// Options from the command line (`amadeus agent`, `make demo`).
#[derive(Default)]
pub struct Launch {
    pub template: Option<String>,
    pub task: Option<String>,
    pub auto: bool,
}

pub struct App {
    pub cfg: Config,
    tx: Sender<AppEvent>,

    pub sb: Scrollback,
    pub scroll: usize,
    pub pinned: bool,

    pub input: String,
    /// Byte offset into `input`.
    pub cursor: usize,
    pub hist: InputHistory,

    pub tier: Tier,
    pub file_context: String,
    pub staged: Vec<String>,
    pub busy: Option<u64>,
    pub busy_since: Option<Instant>,
    pub debug: bool,
    pub health: Option<Health>,

    pub templates: Vec<Template>,
    pub template: Option<Template>,
    /// Messages of the current template conversation, system prompt first.
    conversation: Vec<Value>,
    /// Auto-approve for the next agent run only.
    pub auto_next: bool,
    pub pending: Option<Pending>,
    /// Template picker selection, when open.
    pub picker: Option<usize>,
    pub show_help: bool,
    pub last_log: Option<PathBuf>,
    /// Tool calls in the agent run in flight; None for chat, /rag, /shell.
    run_tools: Option<usize>,

    next_id: u64,
    pub cancel: Arc<AtomicBool>,

    /// Accumulated text of the in-flight chat answer, for the history file.
    response: String,
    last_user: String,

    pub last_key: Instant,
    pub idle_since: Instant,
    pub used: HashSet<String>,
    pub ticks: u64,
    pub should_quit: bool,
}

impl App {
    pub fn new(cfg: Config, tx: Sender<AppEvent>, templates: Vec<Template>) -> App {
        let now = Instant::now();
        App {
            cfg,
            tx,
            sb: Scrollback::default(),
            scroll: 0,
            pinned: true,
            input: String::new(),
            cursor: 0,
            hist: InputHistory::default(),
            tier: Tier::Main,
            file_context: String::new(),
            staged: Vec::new(),
            busy: None,
            busy_since: None,
            debug: false,
            health: None,
            templates,
            template: None,
            conversation: Vec::new(),
            auto_next: false,
            pending: None,
            picker: None,
            show_help: false,
            last_log: None,
            run_tools: None,
            next_id: 0,
            cancel: Arc::new(AtomicBool::new(false)),
            response: String::new(),
            last_user: String::new(),
            last_key: now,
            idle_since: now,
            used: HashSet::new(),
            ticks: 0,
            should_quit: false,
        }
    }

    /// Things that touch the outside world: input history on disk and the
    /// Ollama health check. Kept out of `new` so tests stay hermetic.
    pub fn start(&mut self, launch: Launch, template_errors: Vec<String>) {
        self.hist = InputHistory::load(paths::input_history_file());
        let tx = self.tx.clone();
        let url = self.cfg.ollama_url.clone();
        let tiers = self.cfg.tiers();
        thread::spawn(move || {
            let _ = tx.send(AppEvent::Health(health::check(&url, &tiers)));
        });

        self.welcome();
        for e in template_errors {
            self.push(Kind::Error, format!("template skipped: {e}"));
        }
        if !paths::rag_script().is_file() {
            self.push(
                Kind::Error,
                format!(
                    "AMADEUS checkout not found (expected {}). Run the TUI through amadeus.sh.",
                    paths::rag_script().display()
                ),
            );
        }

        self.auto_next = launch.auto;
        if let Some(name) = launch.template {
            self.use_template(&name);
        }
        if let Some(task) = launch.task {
            self.input = task;
            self.cursor = self.input.len();
            self.submit();
        }
    }

    fn welcome(&mut self) {
        let mut s = String::from(
            "# AMADEUS\n\nA task assistant that runs entirely on this computer. \
             Pick a template, point it at your files, approve each change.\n\n",
        );
        for (i, t) in self.templates.iter().enumerate() {
            s.push_str(&format!(
                "{}. **{}** `/use {}`  \n{}\n",
                i + 1,
                t.title,
                t.name,
                t.description
            ));
        }
        s.push_str("\n`/new` opens the picker · `?` for help · or just type a question to chat");
        self.push(Kind::Agent, s);
    }

    // ---------- scrollback ----------

    pub fn push(&mut self, kind: Kind, text: String) {
        self.sb.push(kind, text);
        self.pinned = true;
    }

    // ---------- request lifecycle ----------

    fn begin(&mut self) -> u64 {
        self.next_id += 1;
        self.busy = Some(self.next_id);
        self.busy_since = Some(Instant::now());
        self.cancel = Arc::new(AtomicBool::new(false));
        self.response.clear();
        self.next_id
    }

    fn end_busy(&mut self) {
        self.run_tools = None;
        self.busy = None;
        self.busy_since = None;
        self.pending = None;
        self.sb.close_live();
        self.response.clear();
    }

    fn finish(&mut self) {
        if let (Some(n), Some(t)) = (self.run_tools, self.busy_since) {
            if !self.cancel.load(Ordering::Relaxed) {
                let noun = if n == 1 { "step" } else { "steps" };
                self.push(
                    Kind::System,
                    format!(
                        "finished in {}s · {n} tool {noun} · /log shows the audit trail",
                        t.elapsed().as_secs()
                    ),
                );
            }
        }
        let answer = render::strip_think(&self.response);
        if !answer.is_empty() && self.template.is_none() {
            self.write_history(&self.last_user.clone(), &answer);
        }
        self.end_busy();
    }

    fn write_history(&self, user: &str, agent: &str) {
        if let Some(dir) = self.cfg.history_file.parent() {
            let _ = fs::create_dir_all(dir);
        }
        if let Ok(mut f) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.cfg.history_file)
        {
            let _ = write!(
                f,
                "[SESSION:{}] User: {}\nAgent: {}\n---\n\n",
                self.cfg.session, user, agent
            );
        }
    }

    fn recent_history(&self) -> String {
        let text = fs::read_to_string(&self.cfg.history_file).unwrap_or_default();
        let lines: Vec<&str> = text.lines().collect();
        let start = lines.len().saturating_sub(40);
        lines[start..].join("\n")
    }

    // ---------- events ----------

    pub fn handle(&mut self, ev: AppEvent) {
        match ev {
            AppEvent::Tick => self.ticks += 1,
            AppEvent::Term(e) => self.on_term(e),
            AppEvent::Health(h) => self.on_health(h),
            AppEvent::Token { id, delta } => {
                if self.busy == Some(id) {
                    if !self.sb.has_live() {
                        self.sb.open_live(Kind::Agent);
                    }
                    self.response.push_str(&delta);
                    self.sb.append_live(&delta);
                }
            }
            AppEvent::TurnEnd { id, keep } => {
                if self.busy == Some(id) {
                    if keep {
                        self.sb.close_live();
                    } else {
                        self.sb.discard_live();
                    }
                }
            }
            AppEvent::Approval {
                id,
                title,
                body,
                reply,
            } => {
                if self.busy == Some(id) {
                    self.sb.close_live();
                    self.pending = Some(Pending {
                        title,
                        body,
                        scroll: 0,
                        note: None,
                        reply,
                    });
                }
            }
            AppEvent::Tool {
                id,
                headline,
                summary,
                ok,
            } => {
                if self.busy == Some(id) {
                    self.sb.close_live();
                    if let Some(n) = self.run_tools.as_mut() {
                        *n += 1;
                    }
                    let mark = if ok { "" } else { "✗ " };
                    self.push(Kind::Tool, format!("{headline}\n{mark}{summary}"));
                }
            }
            AppEvent::RunEnd { id, messages, log } => {
                if self.busy == Some(id) {
                    self.conversation = messages;
                    self.last_log = Some(log);
                }
            }
            AppEvent::Sources {
                id,
                sources,
                candidates,
            } => {
                if self.busy == Some(id) {
                    if !sources.is_empty() {
                        let n = sources.len();
                        let noun = if n == 1 { "passage" } else { "passages" };
                        self.push(
                            Kind::Source,
                            format!("answering from {n} {noun}\n{}", sources.join("\n")),
                        );
                    }
                    if self.debug && !candidates.is_empty() {
                        self.push(
                            Kind::Source,
                            format!(
                                "debug: top-k candidates, cosine distance, before the margin cut\n{}",
                                candidates.join("\n")
                            ),
                        );
                    }
                }
            }
            AppEvent::Note { id, text } => {
                if self.busy == Some(id) || self.busy.is_none() {
                    self.push(Kind::System, text);
                }
            }
            AppEvent::Done { id } => {
                if self.busy == Some(id) {
                    self.finish();
                }
            }
            AppEvent::Failed { id, text } => {
                if self.busy == Some(id) {
                    self.sb.close_live();
                    self.push(Kind::Error, text);
                    self.end_busy();
                }
            }
        }
    }

    fn on_health(&mut self, h: Health) {
        let current = self.cfg.model(self.tier).to_string();
        match &h.version {
            Err(_) => {
                for p in h.problems() {
                    self.push(Kind::Error, p);
                }
            }
            Ok(_) => {
                if h.missing().any(|m| m.name == current) {
                    self.push(
                        Kind::Error,
                        format!("model {current} isn't pulled. Fix: ollama pull {current}"),
                    );
                }
            }
        }
        self.health = Some(h);
    }

    /// Green when Ollama is up and the active model is pulled.
    pub fn ollama_status(&self) -> Option<Status> {
        let h = self.health.as_ref()?;
        Some(match h.status() {
            Status::Down => Status::Down,
            _ if h.missing().any(|m| m.name == self.cfg.model(self.tier)) => Status::Degraded,
            _ => Status::Ok,
        })
    }

    fn on_term(&mut self, e: Event) {
        match e {
            Event::Key(k) => {
                if k.kind != KeyEventKind::Release {
                    self.on_key(k);
                }
            }
            Event::Paste(s) => {
                if let Some(note) = self.pending.as_mut().and_then(|p| p.note.as_mut()) {
                    note.push_str(&s.replace(['\r', '\n'], " "));
                } else if self.pending.is_none() && self.picker.is_none() {
                    self.last_key = Instant::now();
                    self.insert_str(&s.replace("\r\n", "\n").replace('\r', "\n"));
                }
            }
            Event::Mouse(m) => match m.kind {
                MouseEventKind::ScrollUp => self.scroll_up(3),
                MouseEventKind::ScrollDown => self.scroll_down(3),
                _ => {}
            },
            Event::Resize(_, _) => self.sb.invalidate(),
            _ => {}
        }
    }

    fn scroll_up(&mut self, n: usize) {
        self.pinned = false;
        self.scroll = self.scroll.saturating_sub(n);
    }

    fn scroll_down(&mut self, n: usize) {
        // draw() re-pins once this passes the bottom.
        self.scroll += n;
    }

    fn cancel_run(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        // Dropping the pending reply wakes an agent blocked on approval.
        self.pending = None;
        self.push(Kind::System, "cancelling…".into());
    }

    fn on_key(&mut self, k: KeyEvent) {
        self.last_key = Instant::now();
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);

        if ctrl && k.code == KeyCode::Char('c') && self.busy.is_some() {
            self.cancel_run();
            return;
        }
        if self.show_help {
            self.show_help = false;
            return;
        }
        if self.pending.is_some() {
            self.on_approval_key(k);
            return;
        }
        if self.picker.is_some() {
            self.on_picker_key(k);
            return;
        }
        self.on_edit_key(k);
        if self.input.is_empty() {
            self.idle_since = Instant::now();
        }
    }

    fn on_approval_key(&mut self, k: KeyEvent) {
        let Some(p) = self.pending.as_mut() else {
            return;
        };

        // After `n`: an optional one-line reason, sent to the model.
        if let Some(note) = p.note.as_mut() {
            match k.code {
                KeyCode::Enter => {
                    let n = std::mem::take(note);
                    self.decide(Decision::Deny, Some(n));
                }
                KeyCode::Esc => self.decide(Decision::Deny, None),
                KeyCode::Backspace => {
                    note.pop();
                }
                KeyCode::Char(c) => note.push(c),
                _ => {}
            }
            return;
        }

        match k.code {
            KeyCode::PageDown | KeyCode::Down => {
                p.scroll += if k.code == KeyCode::Down { 1 } else { 10 };
            }
            KeyCode::PageUp | KeyCode::Up => {
                p.scroll = p
                    .scroll
                    .saturating_sub(if k.code == KeyCode::Up { 1 } else { 10 });
            }
            KeyCode::Char('n') | KeyCode::Char('N') => p.note = Some(String::new()),
            code => {
                if let Some(d) = Decision::from_key(code) {
                    self.decide(d, None);
                }
            }
        }
    }

    fn decide(&mut self, d: Decision, note: Option<String>) {
        let Some(p) = self.pending.take() else { return };
        let note = note.filter(|n| !n.trim().is_empty());
        let mark = if d.allows() { "✓" } else { "✗" };
        let why = note.as_ref().map(|n| format!(" ({n})")).unwrap_or_default();
        self.push(
            Kind::System,
            format!("{mark} {}: {}{why}", d.label(), p.title),
        );
        let _ = p.reply.send(Verdict { decision: d, note });
    }

    fn on_picker_key(&mut self, k: KeyEvent) {
        let n = self.templates.len();
        let Some(sel) = self.picker else { return };
        match k.code {
            KeyCode::Up => self.picker = Some(sel.checked_sub(1).unwrap_or(n.saturating_sub(1))),
            KeyCode::Down => self.picker = Some(if sel + 1 >= n { 0 } else { sel + 1 }),
            KeyCode::Char(c) if c.is_ascii_digit() => {
                let i = c.to_digit(10).unwrap_or(0) as usize;
                if (1..=n).contains(&i) {
                    self.picker = None;
                    let name = self.templates[i - 1].name.clone();
                    self.use_template(&name);
                }
            }
            KeyCode::Enter => {
                self.picker = None;
                if let Some(t) = self.templates.get(sel) {
                    let name = t.name.clone();
                    self.use_template(&name);
                }
            }
            KeyCode::Esc | KeyCode::Char('q') => self.picker = None,
            _ => {}
        }
    }

    fn on_edit_key(&mut self, k: KeyEvent) {
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        let alt = k.modifiers.contains(KeyModifiers::ALT);

        match k.code {
            KeyCode::Char('c') if ctrl => {
                if self.input.is_empty() {
                    self.should_quit = true;
                } else {
                    self.clear_input();
                }
            }
            KeyCode::Char('d') if ctrl && self.input.is_empty() => self.should_quit = true,
            KeyCode::Char('t') if ctrl => self.cycle_tier(),
            KeyCode::Char('l') if ctrl => self.sb.clear(),
            KeyCode::Char('o') if ctrl => {
                let v = self.sb.view();
                self.sb.set_view(View {
                    show_thinking: !v.show_thinking,
                });
            }
            KeyCode::Char('u') if ctrl => self.clear_input(),
            KeyCode::Char('w') if ctrl => self.delete_word(),
            KeyCode::Char('a') if ctrl => self.cursor = line_start(&self.input, self.cursor),
            KeyCode::Char('e') if ctrl => self.cursor = line_end(&self.input, self.cursor),
            KeyCode::Char('?') if self.input.is_empty() => self.show_help = true,
            KeyCode::Enter if alt => self.insert_str("\n"),
            KeyCode::Enter => self.submit(),
            KeyCode::Tab => self.complete(),
            KeyCode::Backspace => self.backspace(),
            KeyCode::Delete => self.delete(),
            KeyCode::Left => self.cursor = prev_boundary(&self.input, self.cursor),
            KeyCode::Right => self.cursor = next_boundary(&self.input, self.cursor),
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => {
                if self.input.is_empty() {
                    self.pinned = true;
                } else {
                    self.cursor = self.input.len();
                }
            }
            KeyCode::PageUp => self.scroll_up(10),
            KeyCode::PageDown => self.scroll_down(10),
            KeyCode::Up if ctrl => self.scroll_up(1),
            KeyCode::Down if ctrl => self.scroll_down(1),
            KeyCode::Up => self.history_or_line(true),
            KeyCode::Down => self.history_or_line(false),
            KeyCode::Esc => {}
            KeyCode::Char(c) => {
                let s = c.to_string();
                self.insert_str(&s);
            }
            _ => {}
        }
    }

    fn cycle_tier(&mut self) {
        self.tier = self.tier.next();
        let m = self.cfg.model(self.tier).to_string();
        let missing = self
            .health
            .as_ref()
            .is_some_and(|h| h.missing().any(|x| x.name == m));
        let note = if missing {
            format!(" — not pulled, run: ollama pull {m}")
        } else {
            String::new()
        };
        self.push(
            Kind::System,
            format!("model tier → {} ({m}){note}", self.tier.label()),
        );
    }

    /// Up/Down move between lines of a multi-line input, and recall history
    /// when the cursor is already on the first/last line.
    fn history_or_line(&mut self, up: bool) {
        let moved = if up {
            move_up(&self.input, self.cursor)
        } else {
            move_down(&self.input, self.cursor)
        };
        if let Some(c) = moved.filter(|_| !self.hist.browsing()) {
            self.cursor = c;
            return;
        }
        let recalled = if up {
            self.hist.up(&self.input)
        } else {
            self.hist.down()
        };
        if let Some(s) = recalled {
            self.input = s;
            self.cursor = self.input.len();
        }
    }

    // ---------- input editing ----------

    fn insert_str(&mut self, s: &str) {
        self.input.insert_str(self.cursor, s);
        self.cursor += s.len();
    }

    fn clear_input(&mut self) {
        self.input.clear();
        self.cursor = 0;
        self.hist.reset();
        self.idle_since = Instant::now();
    }

    fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let p = prev_boundary(&self.input, self.cursor);
        self.input.replace_range(p..self.cursor, "");
        self.cursor = p;
    }

    fn delete(&mut self) {
        if self.cursor >= self.input.len() {
            return;
        }
        let n = next_boundary(&self.input, self.cursor);
        self.input.replace_range(self.cursor..n, "");
    }

    fn delete_word(&mut self) {
        let head = &self.input[..self.cursor];
        let trimmed = head.trim_end();
        let start = match trimmed.rfind(char::is_whitespace) {
            Some(i) => i + 1,
            None => 0,
        };
        self.input.replace_range(start..self.cursor, "");
        self.cursor = start;
    }

    fn complete(&mut self) {
        if !self.input.starts_with('/') {
            return;
        }
        let first = self
            .input
            .split_whitespace()
            .next()
            .unwrap_or("/")
            .to_string();
        // `/use fo` + tab completes the template name.
        if first == "/use" {
            let partial = self.input["/use".len()..].trim();
            if let Some(t) = self.templates.iter().find(|t| t.name.starts_with(partial)) {
                self.input = format!("/use {}", t.name);
                self.cursor = self.input.len();
            }
            return;
        }
        if self.input.len() != first.len() {
            return;
        }
        if let Some(c) = hint::matches(&first).first() {
            self.input = format!("{} ", c.name);
            self.cursor = self.input.len();
        }
    }

    // ---------- dispatch ----------

    fn submit(&mut self) {
        let text = self.input.trim().to_string();
        if text.is_empty() {
            return;
        }
        if self.busy.is_some() {
            self.push(Kind::Error, "still working — ctrl+c to cancel".into());
            return;
        }
        self.hist.push(&text);
        self.clear_input();
        self.push(Kind::User, text.clone());
        self.route(&text);
    }

    fn route(&mut self, text: &str) {
        if !text.starts_with('/') {
            if self.template.is_some() {
                self.run_task(text.to_string());
            } else {
                self.ask(text.to_string());
            }
            return;
        }

        let (cmd, arg) = match text.find(char::is_whitespace) {
            Some(i) => (&text[..i], text[i..].trim()),
            None => (text, ""),
        };
        self.used.insert(cmd.to_string());

        match cmd {
            "/quit" | "/exit" => self.should_quit = true,
            "/help" => self.show_help = true,
            "/new" => self.open_picker(),
            "/use" => {
                if arg.is_empty() {
                    self.open_picker();
                } else {
                    self.use_template(arg);
                }
            }
            "/chat" => {
                self.template = None;
                self.conversation.clear();
                self.push(Kind::System, "plain chat: no tools, no files".into());
            }
            "/reset" => {
                self.conversation.clear();
                self.push(
                    Kind::System,
                    "conversation cleared; the template stays".into(),
                );
            }
            "/auto" => {
                self.auto_next = !self.auto_next;
                let s = if self.auto_next {
                    "auto-approve ON for the next run: writes and commands will run without asking"
                } else {
                    "auto-approve off: every write and command asks first"
                };
                self.push(Kind::System, s.into());
            }
            "/log" => match &self.last_log {
                Some(p) => {
                    let p = p.display().to_string();
                    self.push(
                        Kind::System,
                        format!("run log: {p}\nreadable: amadeus replay {p}"),
                    );
                }
                None => self.push(Kind::System, "no run yet in this session".into()),
            },
            "/model" => self.cycle_tier(),
            "/debug" => {
                self.debug = !self.debug;
                let s = if self.debug {
                    "on: retrieval distances shown"
                } else {
                    "off"
                };
                self.push(Kind::System, format!("debug {s}"));
            }
            "/clear" => {
                self.sb.clear();
                self.file_context.clear();
                self.staged.clear();
                self.conversation.clear();
                let _ = fs::write(&self.cfg.history_file, "");
                self.push(
                    Kind::System,
                    "screen, conversation and chat history cleared".into(),
                );
            }
            "/history" => {
                let h = fs::read_to_string(&self.cfg.history_file).unwrap_or_default();
                let h = if h.trim().is_empty() {
                    "(empty)".into()
                } else {
                    h
                };
                self.push(Kind::System, h);
            }
            "/file" => self.stage_file(arg),
            "/index" => self.index(arg),
            "/shell" => {
                if arg.is_empty() {
                    self.push(Kind::Error, "usage: /shell <cmd>".into());
                } else {
                    self.shell(arg.to_string());
                }
            }
            "/rag" => {
                if arg.is_empty() {
                    self.push(Kind::Error, "usage: /rag <question>".into());
                } else {
                    self.rag(arg.to_string());
                }
            }
            "/ask" => {
                if arg.is_empty() {
                    self.push(Kind::Error, "usage: /ask <task>".into());
                } else {
                    self.ask(arg.to_string());
                }
            }
            "/agent" => {
                if arg.is_empty() {
                    self.push(Kind::Error, "usage: /agent <task>".into());
                } else if self.use_template("code-helper") {
                    self.input.clear();
                    self.cursor = 0;
                    self.run_task(arg.to_string());
                }
            }
            other => self.push(Kind::Error, format!("unknown command {other} (? for help)")),
        }
    }

    fn open_picker(&mut self) {
        if self.templates.is_empty() {
            self.push(
                Kind::Error,
                "no templates found in AMADEUS_HOME/templates".into(),
            );
            return;
        }
        let cur = self
            .template
            .as_ref()
            .and_then(|t| self.templates.iter().position(|x| x.name == t.name))
            .unwrap_or(0);
        self.picker = Some(cur);
    }

    /// Switch to a template: new conversation, its tier, and its example in
    /// the input box ready to send or edit.
    pub fn use_template(&mut self, name: &str) -> bool {
        let Some(t) = self.templates.iter().find(|t| t.name == name).cloned() else {
            let names: Vec<&str> = self.templates.iter().map(|t| t.name.as_str()).collect();
            self.push(
                Kind::Error,
                format!("no template {name:?}. Available: {}", names.join(", ")),
            );
            return false;
        };
        self.tier = t.tier();
        self.conversation.clear();
        let approval = if t.uses_approval() {
            "you approve every file write and command"
        } else {
            "read-only: it can't change anything"
        };
        self.push(
            Kind::System,
            format!(
                "using {} · {} model · tools: {} · {approval}",
                t.title,
                self.tier.label(),
                t.tools.join(", ")
            ),
        );
        self.input = t.example.clone();
        self.cursor = self.input.len();
        self.template = Some(t);
        true
    }

    fn run_task(&mut self, task: String) {
        let Some(t) = self.template.clone() else {
            return;
        };
        let id = self.begin();
        self.run_tools = Some(0);
        self.last_user = task.clone();
        let auto = std::mem::take(&mut self.auto_next);
        if auto {
            self.push(Kind::System, "auto-approve is on for this run".into());
        }
        let spec = RunSpec {
            url: self.cfg.ollama_url.clone(),
            model: self.cfg.model(self.tier).to_string(),
            template: t,
            root: self.cfg.root.clone(),
            history: self.conversation.clone(),
            task,
            auto,
            log_dir: self.cfg.runs_dir.clone(),
        };
        let (tx, cancel) = (self.tx.clone(), self.cancel.clone());
        thread::spawn(move || agent::run(spec, id, tx, cancel));
    }

    fn stage_file(&mut self, path: &str) {
        if path.is_empty() {
            self.push(Kind::Error, "usage: /file <path>".into());
            return;
        }
        match fs::read_to_string(self.cfg.root.join(path)) {
            Ok(content) => {
                let lines = content.lines().count();
                self.file_context
                    .push_str(&format!("\n\n### File: {path}\n```\n{content}\n```"));
                self.staged.push(path.to_string());
                self.push(
                    Kind::System,
                    format!("staged {path} ({lines} lines) for the next prompt"),
                );
            }
            Err(e) => self.push(Kind::Error, format!("{path}: {e}")),
        }
    }

    fn ask(&mut self, task: String) {
        let user = format!(
            "--- Recent History ---\n{}\n--- End History ---\n{}\n\nTask: {task}",
            self.recent_history(),
            self.file_context
        );
        self.file_context.clear();
        self.staged.clear();
        self.last_user = task;
        self.spawn_chat(vec![
            json!({"role": "system", "content": CHAT_PROMPT}),
            json!({"role": "user", "content": user}),
        ]);
    }

    fn spawn_chat(&mut self, messages: Vec<Value>) {
        let id = self.begin();
        let (cancel, tx) = (self.cancel.clone(), self.tx.clone());
        let url = self.cfg.ollama_url.clone();
        let model = self.cfg.model(self.tier).to_string();
        thread::spawn(move || stream_chat(&url, &model, messages, id, &tx, &cancel));
    }

    fn index(&mut self, arg: &str) {
        let dir = if arg.is_empty() { "." } else { arg }.to_string();
        let id = self.begin();
        let (root, tx, cancel) = (self.cfg.root.clone(), self.tx.clone(), self.cancel.clone());
        thread::spawn(move || {
            let res = rag::index(&root, &dir, &cancel, |l| {
                let _ = tx.send(AppEvent::Note {
                    id,
                    text: l.to_string(),
                });
            });
            let _ = tx.send(match res {
                Ok(()) => AppEvent::Done { id },
                Err(text) => AppEvent::Failed { id, text },
            });
        });
    }

    fn rag(&mut self, question: String) {
        let id = self.begin();
        self.last_user = format!("rag {question}");
        let (cancel, tx) = (self.cancel.clone(), self.tx.clone());
        let url = self.cfg.ollama_url.clone();
        let root = self.cfg.root.clone();
        let model = self.cfg.model(self.tier).to_string();

        thread::spawn(move || match rag::query(&root, &question, &cancel) {
            QueryOutcome::Ok(r) => {
                let _ = tx.send(AppEvent::Sources {
                    id,
                    sources: r.sources,
                    candidates: r.candidates,
                });
                let msgs = vec![json!({"role": "user", "content": r.prompt})];
                stream_chat(&url, &model, msgs, id, &tx, &cancel);
            }
            QueryOutcome::NoContext => {
                let _ = tx.send(AppEvent::Failed {
                    id,
                    text: "nothing relevant in the index for that question".into(),
                });
            }
            QueryOutcome::Cancelled => cancelled(id, &tx),
            QueryOutcome::Err(e) => {
                let _ = tx.send(AppEvent::Failed { id, text: e });
            }
        });
    }

    fn shell(&mut self, cmd: String) {
        let id = self.begin();
        let (cancel, tx) = (self.cancel.clone(), self.tx.clone());
        let root = self.cfg.root.clone();
        let timeout = self.cfg.shell_timeout;
        thread::spawn(move || {
            let mut c = std::process::Command::new("sh");
            c.arg("-c").arg(&cmd).current_dir(&root);
            let res = proc::run(c, &cancel, Some(timeout), |_, l| {
                let _ = tx.send(AppEvent::Note {
                    id,
                    text: l.to_string(),
                });
            });
            finish_proc(id, &tx, res, "shell");
        });
    }

    // ---------- view helpers ----------

    pub fn hint(&self) -> hint::Hint {
        if let Some(p) = &self.pending {
            return hint::Hint::Tip(if p.note.is_some() {
                "type what should change, enter to send · esc skips without a reason".into()
            } else {
                "review the change: y approve · n skip and say why · a approve the rest".into()
            });
        }
        if self.busy.is_some() && self.input.is_empty() {
            return hint::Hint::Tip("working… ctrl+c stops it".into());
        }
        let now = Instant::now();
        hint::hint(
            &self.input,
            now.duration_since(self.last_key),
            now.duration_since(self.idle_since),
            &self.used,
        )
    }

    pub fn spinner(&self) -> &'static str {
        const FRAMES: [&str; 8] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"];
        FRAMES[(self.ticks / 2) as usize % FRAMES.len()]
    }

    pub fn status_right(&self) -> String {
        let mut s = String::new();
        if let Some(t) = self.busy_since {
            if self.pending.is_some() {
                s.push_str("waiting for you · ");
            } else {
                s.push_str(&format!("{} {}s · ", self.spinner(), t.elapsed().as_secs()));
            }
        }
        if self.auto_next {
            s.push_str("auto-approve next run · ");
        }
        if !self.staged.is_empty() {
            s.push_str(&format!("{} staged · ", self.staged.len()));
        }
        s.push_str(&format!(
            "{} {}",
            self.tier.label(),
            self.cfg.model(self.tier)
        ));
        s
    }
}

fn cancelled(id: u64, tx: &Sender<AppEvent>) {
    let _ = tx.send(AppEvent::Note {
        id,
        text: "[cancelled]".into(),
    });
    let _ = tx.send(AppEvent::Done { id });
}

/// Exactly one terminal event for a finished child process.
fn finish_proc(id: u64, tx: &Sender<AppEvent>, res: std::io::Result<proc::Output>, what: &str) {
    match res {
        Ok(o) if o.success() => {
            let _ = tx.send(AppEvent::Done { id });
        }
        Ok(o) if matches!(o.exit, proc::Exit::Cancelled) => cancelled(id, tx),
        Ok(o) => {
            let _ = tx.send(AppEvent::Failed {
                id,
                text: format!("{what}: {}", o.describe()),
            });
        }
        Err(e) => {
            let _ = tx.send(AppEvent::Failed {
                id,
                text: format!("{what} didn't start: {e}"),
            });
        }
    }
}

/// One streamed chat turn, reported as Token events and exactly one terminal
/// event (Done, or Failed).
fn stream_chat(
    url: &str,
    model: &str,
    messages: Vec<Value>,
    id: u64,
    tx: &Sender<AppEvent>,
    cancel: &Arc<AtomicBool>,
) {
    let body = chat_body(url, model, messages, None);
    let res = ollama::chat(url, body, cancel, |d| {
        if let Delta::Content(c) = d {
            let _ = tx.send(AppEvent::Token {
                id,
                delta: c.to_string(),
            });
        }
    });
    match res {
        Ok(_) => {
            let _ = tx.send(AppEvent::Done { id });
        }
        Err(ChatError::Cancelled) => cancelled(id, tx),
        Err(e) => {
            let _ = tx.send(AppEvent::Failed {
                id,
                text: e.to_string(),
            });
        }
    }
}

fn prev_boundary(s: &str, i: usize) -> usize {
    if i == 0 {
        return 0;
    }
    let mut j = i - 1;
    while j > 0 && !s.is_char_boundary(j) {
        j -= 1;
    }
    j
}

fn next_boundary(s: &str, i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    let mut j = i + 1;
    while j < s.len() && !s.is_char_boundary(j) {
        j += 1;
    }
    j
}

fn line_start(s: &str, i: usize) -> usize {
    s[..i].rfind('\n').map_or(0, |p| p + 1)
}

fn line_end(s: &str, i: usize) -> usize {
    s[i..].find('\n').map_or(s.len(), |p| i + p)
}

/// Byte offset `cols` characters into the line starting at `start`, clamped
/// to that line's end.
fn at_col(s: &str, start: usize, cols: usize) -> usize {
    let end = line_end(s, start);
    s[start..end]
        .char_indices()
        .nth(cols)
        .map_or(end, |(o, _)| start + o)
}

fn move_up(s: &str, i: usize) -> Option<usize> {
    let ls = line_start(s, i);
    if ls == 0 {
        return None;
    }
    let col = s[ls..i].chars().count();
    Some(at_col(s, line_start(s, ls - 1), col))
}

fn move_down(s: &str, i: usize) -> Option<usize> {
    let le = line_end(s, i);
    if le == s.len() {
        return None;
    }
    let col = s[line_start(s, i)..i].chars().count();
    Some(at_col(s, le + 1, col))
}

/// Lay the input buffer out into display rows and locate the cursor.
///
/// Hard wrap at `width` so the cursor arithmetic stays exact; the scrollback
/// uses word wrapping instead.
pub fn layout_input(text: &str, cursor: usize, width: usize) -> (Vec<String>, usize, usize) {
    let width = width.max(1);
    let mut rows: Vec<String> = vec![String::new()];
    let mut col = 0usize;
    let (mut crow, mut ccol) = (0usize, 0usize);
    let mut placed = false;

    for (idx, ch) in text.char_indices() {
        if idx == cursor {
            crow = rows.len() - 1;
            ccol = col;
            placed = true;
        }
        if ch == '\n' {
            rows.push(String::new());
            col = 0;
            continue;
        }
        if col == width {
            rows.push(String::new());
            col = 0;
        }
        rows.last_mut().unwrap().push(ch);
        col += 1;
    }
    if !placed {
        crow = rows.len() - 1;
        ccol = col;
    }
    (rows, crow, ccol)
}

pub const QUIET_REDRAW: Duration = Duration::from_millis(250);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::templates;
    use ratatui::crossterm::event::{KeyEventState, MouseEvent};
    use std::sync::mpsc;

    fn tpl(name: &str, tools: &[&str]) -> Template {
        templates::parse(&format!(
            "name = \"{name}\"\ntitle = \"T {name}\"\ndescription = \"d\"\nsystem = \"s\"\n\
             tools = [{}]\nexample = \"example for {name}\"\n",
            tools
                .iter()
                .map(|t| format!("\"{t}\""))
                .collect::<Vec<_>>()
                .join(", ")
        ))
        .unwrap()
    }

    fn app() -> App {
        let (tx, rx) = mpsc::channel();
        // Leak the receiver so sends from worker threads never fail.
        std::mem::forget(rx);
        let mut cfg = Config::for_root(PathBuf::from("/nonexistent/ws"));
        // Hermetic: refused connections, and logs in a temp dir.
        cfg.ollama_url = "http://127.0.0.1:1".into();
        cfg.runs_dir =
            std::env::temp_dir().join(format!("amadeus-test-runs-{}", std::process::id()));
        App::new(
            cfg,
            tx,
            vec![
                tpl("one", &["read_file"]),
                tpl("two", &["read_file", "write_file"]),
            ],
        )
    }

    fn key(code: KeyCode, mods: KeyModifiers) -> AppEvent {
        AppEvent::Term(Event::Key(KeyEvent {
            code,
            modifiers: mods,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        }))
    }

    fn press(a: &mut App, c: char) {
        a.handle(key(KeyCode::Char(c), KeyModifiers::NONE));
    }

    fn texts(a: &App) -> Vec<(Kind, String)> {
        a.sb.entries()
            .iter()
            .map(|e| (e.kind, e.text.clone()))
            .collect()
    }

    #[test]
    fn ctrl_c_mid_stream_does_not_swallow_later_tokens() {
        let mut a = app();
        let id = a.begin();
        a.handle(AppEvent::Token {
            id,
            delta: "Hel".into(),
        });
        a.handle(key(KeyCode::Char('c'), KeyModifiers::CONTROL));
        a.handle(AppEvent::Token {
            id,
            delta: "lo".into(),
        });
        a.handle(AppEvent::Note {
            id,
            text: "[cancelled]".into(),
        });
        a.handle(AppEvent::Done { id });

        let t = texts(&a);
        let n = t.len();
        assert_eq!(t[n - 3], (Kind::Agent, "Hello".into()));
        assert_eq!(t[n - 2], (Kind::System, "cancelling…".into()));
        assert_eq!(t[n - 1], (Kind::System, "[cancelled]".into()));
        assert!(a.cancel.load(Ordering::Relaxed));
        assert!(a.busy.is_none());
    }

    #[test]
    fn ctrl_l_mid_stream_keeps_streaming_into_the_same_answer() {
        let mut a = app();
        let id = a.begin();
        a.handle(AppEvent::Token {
            id,
            delta: "par".into(),
        });
        a.handle(key(KeyCode::Char('l'), KeyModifiers::CONTROL));
        a.handle(AppEvent::Token {
            id,
            delta: "tial".into(),
        });
        assert_eq!(texts(&a), vec![(Kind::Agent, "partial".into())]);
    }

    #[test]
    fn tokens_from_a_stale_request_are_ignored() {
        let mut a = app();
        let old = a.begin();
        a.handle(AppEvent::Done { id: old });
        let new = a.begin();
        a.handle(AppEvent::Token {
            id: old,
            delta: "stale".into(),
        });
        a.handle(AppEvent::Token {
            id: new,
            delta: "fresh".into(),
        });
        assert_eq!(texts(&a).last().unwrap().1, "fresh");
    }

    #[test]
    fn tool_call_text_is_discarded_and_the_next_turn_opens_fresh() {
        let mut a = app();
        let id = a.begin();
        a.handle(AppEvent::Token {
            id,
            delta: "{\"name\":\"read_file\"}".into(),
        });
        a.handle(AppEvent::TurnEnd { id, keep: false });
        a.handle(AppEvent::Tool {
            id,
            headline: "read_file a".into(),
            summary: "3 lines".into(),
            ok: true,
        });
        a.handle(AppEvent::Token {
            id,
            delta: "Summary".into(),
        });
        a.handle(AppEvent::TurnEnd { id, keep: true });
        let t = texts(&a);
        assert!(!t.iter().any(|(_, s)| s.contains("\"name\"")));
        assert_eq!(t[t.len() - 2], (Kind::Tool, "read_file a\n3 lines".into()));
        assert_eq!(t[t.len() - 1], (Kind::Agent, "Summary".into()));
    }

    fn approval(a: &mut App) -> mpsc::Receiver<Verdict> {
        let id = a.begin();
        let (rtx, rrx) = mpsc::channel();
        a.handle(AppEvent::Approval {
            id,
            title: "create x.md".into(),
            body: "+hi".into(),
            reply: rtx,
        });
        rrx
    }

    #[test]
    fn approval_keys_reach_the_agent_and_never_the_input() {
        for (k, want) in [('y', Decision::Approve), ('a', Decision::ApproveRest)] {
            let mut a = app();
            let rrx = approval(&mut a);
            press(&mut a, 'z'); // ignored
            assert!(a.pending.is_some());
            press(&mut a, k);
            assert_eq!(rrx.try_recv().unwrap(), Verdict::of(want));
            assert!(a.pending.is_none());
            assert!(a.input.is_empty(), "approval keys must not be typed");
            assert!(texts(&a).last().unwrap().1.contains("create x.md"));
        }
    }

    #[test]
    fn n_asks_why_and_sends_the_reason() {
        let mut a = app();
        let rrx = approval(&mut a);
        press(&mut a, 'n');
        assert!(
            rrx.try_recv().is_err(),
            "nothing sent until the reason is done"
        );
        for c in "no names".chars() {
            press(&mut a, c);
        }
        a.handle(key(KeyCode::Enter, KeyModifiers::NONE));
        let v = rrx.try_recv().unwrap();
        assert_eq!(v.decision, Decision::Deny);
        assert_eq!(v.note.as_deref(), Some("no names"));
        assert!(a.input.is_empty());
        assert!(texts(&a).last().unwrap().1.ends_with("(no names)"));
    }

    #[test]
    fn esc_declines_without_a_reason() {
        let mut a = app();
        let rrx = approval(&mut a);
        a.handle(key(KeyCode::Esc, KeyModifiers::NONE));
        assert_eq!(rrx.try_recv().unwrap(), Verdict::of(Decision::Deny));
        let rrx = approval(&mut a);
        press(&mut a, 'n');
        a.handle(key(KeyCode::Esc, KeyModifiers::NONE));
        assert_eq!(rrx.try_recv().unwrap(), Verdict::of(Decision::Deny));
    }

    #[test]
    fn enter_does_not_approve() {
        let mut a = app();
        let id = a.begin();
        let (rtx, rrx) = mpsc::channel();
        a.handle(AppEvent::Approval {
            id,
            title: "t".into(),
            body: String::new(),
            reply: rtx,
        });
        a.handle(key(KeyCode::Enter, KeyModifiers::NONE));
        assert!(rrx.try_recv().is_err());
        assert!(a.pending.is_some());
    }

    #[test]
    fn ctrl_c_during_approval_cancels_and_releases_the_agent() {
        let mut a = app();
        let id = a.begin();
        let (rtx, rrx) = mpsc::channel();
        a.handle(AppEvent::Approval {
            id,
            title: "t".into(),
            body: String::new(),
            reply: rtx,
        });
        a.handle(key(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(a.pending.is_none());
        assert!(a.cancel.load(Ordering::Relaxed));
        assert!(matches!(
            rrx.try_recv(),
            Err(mpsc::TryRecvError::Disconnected)
        ));
    }

    #[test]
    fn picker_selects_a_template_and_prefills_its_example() {
        let mut a = app();
        a.input = "/new".into();
        a.cursor = 4;
        a.handle(key(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(a.picker, Some(0));
        a.handle(key(KeyCode::Down, KeyModifiers::NONE));
        a.handle(key(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(a.template.as_ref().unwrap().name, "two");
        assert_eq!(a.input, "example for two");
        assert!(texts(&a)
            .last()
            .unwrap()
            .1
            .contains("you approve every file write"));
    }

    #[test]
    fn use_unknown_template_lists_the_real_ones() {
        let mut a = app();
        assert!(!a.use_template("nope"));
        assert!(texts(&a).last().unwrap().1.contains("one, two"));
    }

    #[test]
    fn auto_approve_is_consumed_by_one_run() {
        let mut a = app();
        a.use_template("one");
        a.auto_next = true;
        a.input = "go".into();
        a.cursor = 2;
        a.handle(key(KeyCode::Enter, KeyModifiers::NONE));
        assert!(!a.auto_next, "the next run asks again");
        assert!(a.busy.is_some());
    }

    #[test]
    fn question_mark_opens_help_only_on_an_empty_input() {
        let mut a = app();
        press(&mut a, '?');
        assert!(a.show_help);
        press(&mut a, 'x');
        assert!(!a.show_help);
        assert!(a.input.is_empty(), "the closing key isn't typed");
        press(&mut a, 'w');
        press(&mut a, '?');
        assert_eq!(a.input, "w?");
    }

    #[test]
    fn sources_show_paths_and_hide_distances_unless_debug() {
        let mut a = app();
        let id = a.begin();
        a.handle(AppEvent::Sources {
            id,
            sources: vec!["a.go:1-9".into()],
            candidates: vec!["0.4100  a.go:1-9".into()],
        });
        let t = texts(&a);
        assert_eq!(t.last().unwrap().1, "answering from 1 passage\na.go:1-9");
        assert!(!t
            .iter()
            .any(|(_, s)| s.contains("0.4100") || s.contains("pre-rerank")));

        a.debug = true;
        a.handle(AppEvent::Sources {
            id,
            sources: vec![],
            candidates: vec!["0.4100  a.go:1-9".into()],
        });
        assert!(texts(&a).last().unwrap().1.contains("0.4100"));
    }

    #[test]
    fn mouse_wheel_unpins_and_scrolls() {
        let mut a = app();
        a.scroll = 20;
        let wheel = |kind| {
            AppEvent::Term(Event::Mouse(MouseEvent {
                kind,
                column: 0,
                row: 0,
                modifiers: KeyModifiers::NONE,
            }))
        };
        a.handle(wheel(MouseEventKind::ScrollUp));
        assert!(!a.pinned);
        assert_eq!(a.scroll, 17);
        a.handle(wheel(MouseEventKind::ScrollDown));
        assert_eq!(a.scroll, 20);
    }

    #[test]
    fn up_recalls_history_and_down_restores_the_draft() {
        let mut a = app();
        a.hist.push("earlier task");
        a.input = "draft".into();
        a.cursor = 5;
        a.handle(key(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(a.input, "earlier task");
        a.handle(key(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(a.input, "draft");
    }

    #[test]
    fn up_moves_within_multiline_input_before_recalling() {
        let mut a = app();
        a.hist.push("old");
        a.input = "ab\ncd".into();
        a.cursor = 4; // after 'c'
        a.handle(key(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!((a.input.as_str(), a.cursor), ("ab\ncd", 1));
        a.handle(key(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(a.input, "old");
    }

    #[test]
    fn history_file_lives_in_the_state_dir() {
        let a = app();
        assert!(!a.cfg.history_file.starts_with("/nonexistent/ws"));
    }

    #[test]
    fn cursor_lands_on_the_right_row_after_a_newline() {
        let (rows, r, c) = layout_input("ab\ncd", 4, 20);
        assert_eq!(rows, vec!["ab".to_string(), "cd".to_string()]);
        assert_eq!((r, c), (1, 1));
    }

    #[test]
    fn hard_wrap_splits_at_width() {
        let (rows, _, _) = layout_input("abcdef", 0, 3);
        assert_eq!(rows, vec!["abc".to_string(), "def".to_string()]);
    }

    #[test]
    fn cursor_at_end_is_past_the_last_char() {
        let (_, r, c) = layout_input("hi", 2, 20);
        assert_eq!((r, c), (0, 2));
    }

    #[test]
    fn multibyte_navigation_never_splits_a_char() {
        let s = "héllo";
        let one = next_boundary(s, 0);
        let two = next_boundary(s, one);
        assert_eq!(&s[..two], "hé");
        assert_eq!(prev_boundary(s, two), one);
    }

    #[test]
    fn vertical_moves_clamp_to_shorter_lines() {
        let s = "long line\nab\nxyz";
        assert_eq!(move_down(s, 7), Some(12)); // col 7 → end of "ab"
        assert_eq!(move_up(s, 12), Some(2));
        assert_eq!(move_up(s, 3), None);
        assert_eq!(move_down(s, 14), None);
    }
}
