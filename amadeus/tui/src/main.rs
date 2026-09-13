mod app;
mod event;
mod hint;
mod ollama;
mod rag;
mod ui;

use std::io::{self, Stdout};
use std::sync::mpsc::{self, RecvTimeoutError};

use anyhow::Result;
use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::{DisableBracketedPaste, EnableBracketedPaste};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::Terminal;

use app::{App, Config, QUIET_REDRAW};
use event::AppEvent;

fn main() -> Result<()> {
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
    Ok(Terminal::new(CrosstermBackend::new(out))?)
}

fn restore() {
    let _ = disable_raw_mode();
    let _ = execute!(io::stdout(), DisableBracketedPaste, LeaveAlternateScreen);
}

fn run(terminal: &mut Terminal<CrosstermBackend<Stdout>>, cfg: Config) -> Result<()> {
    let (tx, rx) = mpsc::channel::<AppEvent>();
    event::spawn_input(tx.clone());
    event::spawn_ticker(tx.clone());

    let mut app = App::new(cfg, tx);

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
