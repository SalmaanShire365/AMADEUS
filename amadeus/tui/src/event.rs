use std::path::PathBuf;
use std::sync::mpsc::Sender;
use std::thread;
use std::time::Duration;

use ratatui::crossterm::event::{self, Event};
use serde_json::Value;

use crate::approval::Verdict;
use crate::health::Health;

/// Everything that can wake the main loop. The UI thread never blocks on
/// anything except this channel.
#[derive(Debug)]
pub enum AppEvent {
    /// A key, paste, mouse or resize event from the terminal.
    Term(Event),
    /// Fires every 100ms. Drives the hint rotation and the busy spinner.
    Tick,
    /// The startup health check finished.
    Health(Health),
    /// Retrieval for request `id`: chunks used, and raw candidates for /debug.
    Sources {
        id: u64,
        sources: Vec<String>,
        candidates: Vec<String>,
    },
    /// A streamed token for request `id`.
    Token { id: u64, delta: String },
    /// A model turn ended. `keep` is false when the streamed text was a tool
    /// call written as JSON, which shouldn't stay on screen.
    TurnEnd { id: u64, keep: bool },
    /// The agent wants to run a side-effecting tool and is blocked until a
    /// decision is sent on `reply`.
    Approval {
        id: u64,
        title: String,
        body: String,
        reply: Sender<Verdict>,
    },
    /// A tool call finished (or was declined).
    Tool {
        id: u64,
        headline: String,
        summary: String,
        ok: bool,
    },
    /// An agent run ended: the conversation so far, and where its log is.
    RunEnd {
        id: u64,
        messages: Vec<Value>,
        log: PathBuf,
    },
    /// Request `id` finished normally.
    Done { id: u64 },
    /// A line of plain output (indexing progress, shell output, notices).
    Note { id: u64, text: String },
    /// Request `id` failed. Ends the request.
    Failed { id: u64, text: String },
}

/// Blocking reader for terminal events. No polling, so it costs nothing while
/// the user is idle.
pub fn spawn_input(tx: Sender<AppEvent>) {
    thread::spawn(move || loop {
        match event::read() {
            Ok(ev) => {
                if tx.send(AppEvent::Term(ev)).is_err() {
                    return;
                }
            }
            Err(_) => return,
        }
    });
}

pub fn spawn_ticker(tx: Sender<AppEvent>) {
    thread::spawn(move || loop {
        thread::sleep(Duration::from_millis(100));
        if tx.send(AppEvent::Tick).is_err() {
            return;
        }
    });
}
