// Temporary while the harness lands: tools, templates and doctor use these.
#![allow(dead_code)]

mod app;
mod cli;
mod event;
mod health;
mod hint;
mod history;
mod ollama;
mod paths;
mod proc;
mod rag;
mod render;
mod scrollback;
mod theme;
mod ui;

use std::io::{self, Stdout};
use std::sync::mpsc::{self, RecvTimeoutError};

use anyhow::Result;
use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::Terminal;

use app::{App, Config, QUIET_REDRAW};
use event::AppEvent;

/// Mouse capture gives wheel scrolling but takes over click-drag selection
/// (most terminals still select with shift held). AMADEUS_MOUSE=0 opts out.
fn mouse_enabled() -> bool {
    std::env::var("AMADEUS_MOUSE").map_or(true, |v| v != "0")
}

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("ask") => std::process::exit(cli::ask(args.split_off(1))),
        Some("rag") => std::process::exit(cli::rag(args.split_off(1))),
        _ => {}
    }

    let cfg = Config::from_env();

    // Without this, a panic leaves the terminal in raw mode on the alternate
    // screen and the shell is unusable afterwards.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore();
        default_hook(info);
    }));

    let mut terminal = setup()?;
    let result = run(&mut terminal, cfg);
    restore();
    result
}

fn setup() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    let mut out = io::stdout();
    execute!(out, EnterAlternateScreen, EnableBracketedPaste)?;
    if mouse_enabled() {
        execute!(out, EnableMouseCapture)?;
    }
    Ok(Terminal::new(CrosstermBackend::new(out))?)
}

fn restore() {
    let _ = disable_raw_mode();
    let _ = execute!(
        io::stdout(),
        DisableMouseCapture,
        DisableBracketedPaste,
        LeaveAlternateScreen
    );
}

fn run(terminal: &mut Terminal<CrosstermBackend<Stdout>>, cfg: Config) -> Result<()> {
    let (tx, rx) = mpsc::channel::<AppEvent>();
    event::spawn_input(tx.clone());
    event::spawn_ticker(tx.clone());

    let mut app = App::new(cfg, tx);
    app.start();

    while !app.should_quit {
        terminal.draw(|f| ui::draw(f, &mut app))?;

        match rx.recv_timeout(QUIET_REDRAW) {
            Ok(ev) => {
                app.handle(ev);
                // Drain whatever else is queued before redrawing. Without this
                // a fast token stream costs one full frame per token.
                while let Ok(ev) = rx.try_recv() {
                    app.handle(ev);
                    if app.should_quit {
                        break;
                    }
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }

    Ok(())
}
