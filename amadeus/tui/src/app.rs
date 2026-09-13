use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ratatui::crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
use ratatui::text::Line;

use crate::event::AppEvent;
use crate::hint;
use crate::ollama;
use crate::rag::{self, QueryOutcome};

const SYSTEM_PROMPT: &str = "You are AMADEUS, an expert software engineering agent.
You write clean, production-ready code. When asked to build something,
output complete files with no placeholders. Think step by step.
Be concise — avoid unnecessary explanation unless asked.";

#[derive(Clone, Copy, PartialEq, Eq)]
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
    /// The project AMADEUS is currently operating on.
    pub root: PathBuf,

    pub ollama_url: String,
    pub fast_model: String,
    pub main_model: String,
    pub heavy_model: String,
    pub history_file: PathBuf,
    pub session: u64,
}

impl Config {
    pub fn from_env() -> Config {
        let root = discover_root();

        Config {
            root: root.clone(),

            ollama_url: env_or("AMADEUS_OLLAMA_URL", "http://localhost:11434"),

            fast_model: env_or("AMADEUS_FAST_MODEL", "qwen2.5-coder:1.5b"),

            main_model: env_or("AMADEUS_MAIN_MODEL", "qwen2.5-coder:3b"),

            heavy_model: env_or("AMADEUS_HEAVY_MODEL", "qwen3:8b"),

            history_file: root.join(".agent_history.txt"),

            session: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    pub fn model(&self, tier: Tier) -> &str {
        match tier {
            Tier::Fast => &self.fast_model,
            Tier::Main => &self.main_model,
            Tier::Heavy => &self.heavy_model,
        }
    }

    pub fn rag_script_exists(&self) -> bool {
        rag::script().is_file()
    }
}

/// The current working directory is the user's project.
///
/// IMPORTANT:
/// `AMADEUS_HOME` points to the AMADEUS installation, not the project.
///
/// Example:
///
///   AMADEUS_HOME = ~/gt/amadeus/crew/salmaan/amadeus
///   root         = ~/black-box
///
/// This separation is what allows:
///
///   cd ~/black-box
///   amadeus
///
/// to operate on black-box while still using AMADEUS's own Python runtime,
/// RAG implementation, and agent.py.
fn discover_root() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    User,
    Agent,
    System,
    Source,
    Error,
}

pub struct Entry {
    pub kind: Kind,
    pub text: String,
}

pub struct App {
    pub cfg: Config,
    tx: Sender<AppEvent>,

    pub entries: Vec<Entry>,

    /// Wrapped lines for every entry except the last one.
    pub stable: Vec<Line<'static>>,
    pub stable_upto: usize,
    pub wrap_width: u16,
    pub scroll: usize,
    pub pinned: bool,

    pub input: String,

    /// Byte offset into `input`.
    pub cursor: usize,

    pub tier: Tier,
    pub file_context: String,
    pub staged: Vec<String>,
    pub busy: Option<u64>,

    next_id: u64,
    pub cancel: Arc<AtomicBool>,

    /// Accumulated text of the in-flight answer, for the history file.
    response: String,
    last_user: String,

    /// Index of the Agent entry currently being streamed into.
    agent_idx: Option<usize>,

    pub last_key: Instant,
    pub idle_since: Instant,
    pub used: HashSet<String>,
    pub ticks: u64,
    pub should_quit: bool,
}

impl App {
    pub fn new(cfg: Config, tx: Sender<AppEvent>) -> App {
        let now = Instant::now();

        let mut app = App {
            cfg,
            tx,
            entries: Vec::new(),
            stable: Vec::new(),
            stable_upto: 0,
            wrap_width: 0,
            scroll: 0,
            pinned: true,
            input: String::new(),
            cursor: 0,
            tier: Tier::Main,
            file_context: String::new(),
            staged: Vec::new(),
            busy: None,
            next_id: 0,
            cancel: Arc::new(AtomicBool::new(false)),
            response: String::new(),
            last_user: String::new(),
            agent_idx: None,
            last_key: now,
            idle_since: now,
            used: HashSet::new(),
            ticks: 0,
            should_quit: false,
        };

        let root = app.cfg.root.display().to_string();

        app.push(
            Kind::System,
            format!("AMADEUS — session {}", app.cfg.session),
        );

        app.push(Kind::System, format!("root {root}"));

        if !app.cfg.rag_script_exists() {
            app.push(
                Kind::Error,
                format!(
                    "AMADEUS RAG script not found — expected {}. \
set AMADEUS_HOME to your AMADEUS checkout.",
                    rag::script().display()
                ),
            );
        }

        app.push(
            Kind::System,
            "type / for commands, ctrl+t for model tier".into(),
        );

        app
    }

    // ---------- scrollback ----------

    pub fn push(&mut self, kind: Kind, text: String) {
        self.entries.push(Entry { kind, text });
        self.pin_bottom();
    }

    fn pin_bottom(&mut self) {
        self.pinned = true;
    }

    /// Only the last entry is ever mutated, which is what lets the wrap cache
    /// stay valid for everything before it.
    fn append_last(&mut self, delta: &str) {
        if let Some(e) = self.entries.last_mut() {
            e.text.push_str(delta);
        }
    }

    pub fn invalidate_wrap(&mut self) {
        self.stable.clear();
        self.stable_upto = 0;
    }

    // ---------- request lifecycle ----------

    fn begin(&mut self) -> u64 {
        self.next_id += 1;
        self.busy = Some(self.next_id);
        self.cancel = Arc::new(AtomicBool::new(false));
        self.response.clear();
        self.agent_idx = None;
        self.next_id
    }

    fn open_agent_entry(&mut self) {
        self.entries.push(Entry {
            kind: Kind::Agent,
            text: String::new(),
        });

        self.agent_idx = Some(self.entries.len() - 1);
        self.pin_bottom();
    }

    fn finish(&mut self) {
        self.busy = None;
        self.agent_idx = None;

        if !self.response.trim().is_empty() {
            self.write_history(&self.last_user.clone(), &self.response.clone());
        }

        self.response.clear();
    }

    fn write_history(&self, user: &str, agent: &str) {
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
            AppEvent::Tick => {
                self.ticks += 1;
            }

            AppEvent::Term(e) => {
                self.on_term(e);
            }

            AppEvent::Token { id, delta } => {
                if self.busy == Some(id) {
                    if self.agent_idx.is_none() {
                        self.open_agent_entry();
                    }

                    self.response.push_str(&delta);
                    self.append_last(&delta);
                }
            }

            AppEvent::Sources { id, hits } => {
                if self.busy == Some(id) && !hits.is_empty() {
                    let body = hits.join("\n");

                    self.push(
                        Kind::Source,
                        format!("top-k candidates (pre-rerank, pre-margin)\n{body}"),
                    );
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
                    self.push(Kind::Error, text);
                    self.busy = None;
                    self.agent_idx = None;
                    self.response.clear();
                }
            }
        }
    }

    fn on_term(&mut self, e: Event) {
        match e {
            Event::Key(k) => {
                if k.kind == ratatui::crossterm::event::KeyEventKind::Release {
                    return;
                }

                self.on_key(k);
            }

            Event::Paste(s) => {
                self.last_key = Instant::now();
                self.insert_str(&s);
            }

            Event::Resize(_, _) => {
                self.invalidate_wrap();
            }

            _ => {}
        }
    }

    fn on_key(&mut self, k: KeyEvent) {
        self.last_key = Instant::now();

        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        let alt = k.modifiers.contains(KeyModifiers::ALT);

        match k.code {
            KeyCode::Char('c') if ctrl => {
                if self.busy.is_some() {
                    self.cancel.store(true, Ordering::Relaxed);
                    self.push(Kind::System, "cancelling…".into());
                } else if self.input.is_empty() {
                    self.should_quit = true;
                } else {
                    self.clear_input();
                }
            }

            KeyCode::Char('d') if ctrl && self.input.is_empty() => {
                self.should_quit = true;
            }

            KeyCode::Char('t') if ctrl => {
                self.tier = self.tier.next();

                let m = self.cfg.model(self.tier).to_string();

                self.push(
                    Kind::System,
                    format!("model tier → {} ({m})", self.tier.label()),
                );
            }

            KeyCode::Char('l') if ctrl => {
                self.entries.clear();
                self.invalidate_wrap();
            }

            KeyCode::Char('u') if ctrl => {
                self.clear_input();
            }

            KeyCode::Char('w') if ctrl => {
                self.delete_word();
            }

            KeyCode::Char('a') if ctrl => {
                self.cursor = 0;
            }

            KeyCode::Char('e') if ctrl => {
                self.cursor = self.input.len();
            }

            KeyCode::Enter if alt => {
                self.insert_str("\n");
            }

            KeyCode::Enter => {
                self.submit();
            }

            KeyCode::Tab => {
                self.complete();
            }

            KeyCode::Backspace => {
                self.backspace();
            }

            KeyCode::Delete => {
                self.delete();
            }

            KeyCode::Left => {
                self.cursor = prev_boundary(&self.input, self.cursor);
            }

            KeyCode::Right => {
                self.cursor = next_boundary(&self.input, self.cursor);
            }

            KeyCode::Home => {
                self.cursor = 0;
            }

            KeyCode::End => {
                if self.input.is_empty() {
                    self.pinned = true;
                } else {
                    self.cursor = self.input.len();
                }
            }

            KeyCode::PageUp => {
                self.pinned = false;
                self.scroll = self.scroll.saturating_sub(10);
            }

            KeyCode::PageDown => {
                self.scroll += 10;
            }

            KeyCode::Up if ctrl => {
                self.pinned = false;
                self.scroll = self.scroll.saturating_sub(1);
            }

            KeyCode::Down if ctrl => {
                self.scroll += 1;
            }

            KeyCode::Char(c) => {
                let s = c.to_string();
                self.insert_str(&s);
            }

            _ => {}
        }

        if self.input.is_empty() {
            self.idle_since = Instant::now();
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

        let token = self
            .input
            .split_whitespace()
            .next()
            .unwrap_or("/")
            .to_string();

        if self.input.len() != token.len() {
            return;
        }

        let m = hint::matches(&token);

        if let Some(first) = m.first() {
            self.input = format!("{} ", first.name);
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

        self.clear_input();
        self.push(Kind::User, text.clone());
        self.route(&text);
    }

    fn route(&mut self, text: &str) {
        if !text.starts_with('/') {
            self.ask(text.to_string());
            return;
        }

        let (cmd, arg) = match text.find(char::is_whitespace) {
            Some(i) => (&text[..i], text[i..].trim()),
            None => (text, ""),
        };

        self.used.insert(cmd.to_string());

        match cmd {
            "/quit" | "/exit" => {
                self.should_quit = true;
            }

            "/model" => {
                self.tier = self.tier.next();

                let m = self.cfg.model(self.tier).to_string();

                self.push(
                    Kind::System,
                    format!("model tier → {} ({m})", self.tier.label()),
                );
            }

            "/clear" => {
                self.entries.clear();
                self.invalidate_wrap();
                self.file_context.clear();
                self.staged.clear();

                let _ = fs::write(&self.cfg.history_file, "");

                self.push(Kind::System, "history cleared".into());
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

            "/file" => {
                self.stage_file(arg);
            }

            "/index" => {
                let dir = if arg.is_empty() {
                    ".".to_string()
                } else {
                    arg.to_string()
                };

                let id = self.begin();
                let root = self.cfg.root.clone();
                let tx = self.tx.clone();

                thread::spawn(move || rag::index(&root, &dir, id, &tx));
            }

            "/shell" => {
                if arg.is_empty() {
                    self.push(Kind::Error, "usage: /shell <cmd>".into());
                    return;
                }

                let id = self.begin();
                let cmd = arg.to_string();
                let tx = self.tx.clone();

                thread::spawn(move || rag::shell(&cmd, id, &tx));
            }

            "/rag" => {
                if arg.is_empty() {
                    self.push(Kind::Error, "usage: /rag <question>".into());
                    return;
                }

                self.rag(arg.to_string());
            }

            "/ask" => {
                if arg.is_empty() {
                    self.push(Kind::Error, "usage: /ask <task>".into());
                    return;
                }

                self.ask(arg.to_string());
            }

            "/agent" => {
                if arg.is_empty() {
                    self.push(Kind::Error, "usage: /agent <task>".into());
                    return;
                }

                self.agent(arg.to_string());
            }

            other => {
                self.push(Kind::Error, format!("unknown command {other}"));
            }
        }
    }

    fn stage_file(&mut self, path: &str) {
        if path.is_empty() {
            self.push(Kind::Error, "usage: /file <path>".into());
            return;
        }

        match fs::read_to_string(path) {
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

            Err(e) => {
                self.push(Kind::Error, format!("{path}: {e}"));
            }
        }
    }

    fn ask(&mut self, task: String) {
        let prompt = format!(
            "{SYSTEM_PROMPT}\n\n--- Recent History ---\n{}\n--- End History ---\n{}\n\nTask: {task}",
            self.recent_history(),
            self.file_context
        );

        self.file_context.clear();
        self.staged.clear();
        self.last_user = task;

        self.spawn_generate(prompt);
    }

    fn rag(&mut self, question: String) {
        let id = self.begin();

        self.last_user = format!("rag {question}");

        let cancel = self.cancel.clone();
        let tx = self.tx.clone();
        let url = self.cfg.ollama_url.clone();
        let root = self.cfg.root.clone();
        let model = self.cfg.model(self.tier).to_string();

        thread::spawn(move || match rag::query(&root, &question) {
            QueryOutcome::Ok(r) => {
                let _ = tx.send(AppEvent::Sources { id, hits: r.hits });

                if cancel.load(Ordering::Relaxed) {
                    let _ = tx.send(AppEvent::Done { id });
                    return;
                }

                ollama::generate(&url, &model, &r.prompt, id, &tx, &cancel);
            }

            QueryOutcome::NoContext => {
                let _ = tx.send(AppEvent::Failed {
                    id,
                    text: "nothing relevant in the index for that question".into(),
                });
            }

            QueryOutcome::Err(e) => {
                let _ = tx.send(AppEvent::Failed { id, text: e });
            }
        });
    }

    fn agent(&mut self, task: String) {
        let id = self.begin();

        self.last_user = format!("agent {task}");

        let tx = self.tx.clone();
        let cancel = self.cancel.clone();
        let root = self.cfg.root.clone();
        let model = self.cfg.model(self.tier).to_string();

        thread::spawn(move || {
            /*
             * AMADEUS's own Python runtime.
             *
             * This comes from AMADEUS_HOME, NOT the target project.
             */
            let python = rag::python();

            /*
             * AMADEUS's own agent implementation.
             */
            let script = rag::amadeus_home().join("agent/agent.py");

            if !python.is_file() {
                let _ = tx.send(AppEvent::Failed {
                    id,
                    text: format!("agent Python interpreter not found: {}", python.display()),
                });

                return;
            }

            if !script.is_file() {
                let _ = tx.send(AppEvent::Failed {
                    id,
                    text: format!("agent.py not found: {}", script.display()),
                });

                return;
            }

            let _ = tx.send(AppEvent::Note {
                id,
                text: format!("starting AMADEUS agent with {model} in {}", root.display()),
            });

            let mut child = match std::process::Command::new(&python)
                .arg(&script)
                .arg(&task)
                .arg("--model")
                .arg(&model)
                .arg("--yes")
                /*
                 * The agent operates on the user's project.
                 *
                 * Therefore:
                 *
                 *   Path.cwd()
                 *
                 * inside agent.py becomes the target project.
                 */
                .current_dir(&root)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
            {
                Ok(child) => child,

                Err(e) => {
                    let _ = tx.send(AppEvent::Failed {
                        id,
                        text: format!("failed to start agent: {e}"),
                    });

                    return;
                }
            };

            /*
             * Take both pipes immediately.
             */
            let stdout = child.stdout.take();
            let stderr = child.stderr.take();

            /*
             * IMPORTANT:
             *
             * stdout and stderr are consumed on SEPARATE threads.
             *
             * Do NOT read all of stdout and then all of stderr.
             *
             * If the Python process writes enough stderr to fill the OS
             * pipe buffer while Rust is waiting on stdout, the child can
             * deadlock.
             */

            let tx_stdout = tx.clone();

            let stdout_thread = thread::spawn(move || {
                if let Some(stdout) = stdout {
                    use std::io::{BufRead, BufReader};

                    let reader = BufReader::new(stdout);

                    for line in reader.lines() {
                        match line {
                            Ok(line) => {
                                if !line.trim().is_empty() {
                                    let _ = tx_stdout.send(AppEvent::Note { id, text: line });
                                }
                            }

                            Err(e) => {
                                let _ = tx_stdout.send(AppEvent::Note {
                                    id,
                                    text: format!("agent stdout error: {e}"),
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
                    use std::io::{BufRead, BufReader};

                    let reader = BufReader::new(stderr);

                    for line in reader.lines() {
                        match line {
                            Ok(line) => {
                                if !line.trim().is_empty() {
                                    let _ = tx_stderr.send(AppEvent::Note { id, text: line });
                                }
                            }

                            Err(e) => {
                                let _ = tx_stderr.send(AppEvent::Note {
                                    id,
                                    text: format!("agent stderr error: {e}"),
                                });

                                break;
                            }
                        }
                    }
                }
            });

            /*
             * Wait for the process while still allowing cancellation.
             *
             * The stdout/stderr reader threads drain their respective pipes
             * concurrently, so the Python process cannot block because one
             * pipe is not being consumed.
             */
            loop {
                if cancel.load(Ordering::Relaxed) {
                    let _ = child.kill();
                    let _ = child.wait();

                    let _ = stdout_thread.join();
                    let _ = stderr_thread.join();

                    let _ = tx.send(AppEvent::Note {
                        id,
                        text: "agent cancelled".into(),
                    });

                    let _ = tx.send(AppEvent::Done { id });

                    return;
                }

                match child.try_wait() {
                    Ok(Some(status)) => {
                        /*
                         * The process has exited. Its pipe threads will finish
                         * after consuming any remaining buffered output.
                         */
                        let _ = stdout_thread.join();
                        let _ = stderr_thread.join();

                        if status.success() {
                            let _ = tx.send(AppEvent::Done { id });
                        } else {
                            let _ = tx.send(AppEvent::Failed {
                                id,
                                text: format!("agent exited with {status}"),
                            });
                        }

                        return;
                    }

                    Ok(None) => {
                        thread::sleep(Duration::from_millis(50));
                    }

                    Err(e) => {
                        let _ = child.kill();
                        let _ = child.wait();

                        let _ = stdout_thread.join();
                        let _ = stderr_thread.join();

                        let _ = tx.send(AppEvent::Failed {
                            id,
                            text: format!("failed checking agent process: {e}"),
                        });

                        return;
                    }
                }
            }
        });
    }

    fn spawn_generate(&mut self, prompt: String) {
        let id = self.begin();

        let cancel = self.cancel.clone();
        let tx = self.tx.clone();
        let url = self.cfg.ollama_url.clone();
        let model = self.cfg.model(self.tier).to_string();

        thread::spawn(move || {
            ollama::generate(&url, &model, &prompt, id, &tx, &cancel);
        });
    }

    // ---------- view helpers ----------

    pub fn hint(&self) -> hint::Hint {
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

        if self.busy.is_some() {
            s.push_str(self.spinner());
            s.push(' ');
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
}
