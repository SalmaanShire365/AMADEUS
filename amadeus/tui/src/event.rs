use std::sync::mpsc::Sender;
use std::thread;
use std::time::Duration;

use ratatui::crossterm::event::{self, Event};

/// Everything that can wake the main loop. The UI thread never blocks on
/// anything except this channel.
#[derive(Debug)]
pub enum AppEvent {
    /// A key, paste or resize from the terminal.
    Term(Event),
    /// Fires every 100ms. Drives the hint rotation and the busy spinner.
    Tick,
    /// Retrieval candidates for request `id`, already formatted for display.
    Sources { id: u64, hits: Vec<String> },
    /// A streamed token from Ollama.
    Token { id: u64, delta: String },
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
