//! One palette for the whole UI.
//!
//! Only the eight basic ANSI hues plus modifiers, and never White, Black, Gray
//! or Yellow as a foreground: those are the ones that vanish on either a light
//! or a dark background. Secondary text uses DIM rather than DarkGray so it
//! tracks whatever the terminal's foreground is.

use ratatui::style::{Color, Modifier, Style};

pub fn accent() -> Style {
    Style::default().fg(Color::Cyan)
}

pub fn accent_bold() -> Style {
    accent().add_modifier(Modifier::BOLD)
}

pub fn dim() -> Style {
    Style::default().add_modifier(Modifier::DIM)
}

pub fn dim_italic() -> Style {
    dim().add_modifier(Modifier::ITALIC)
}

pub fn bold() -> Style {
    Style::default().add_modifier(Modifier::BOLD)
}

pub fn code() -> Style {
    Style::default().fg(Color::Green)
}

pub fn tool() -> Style {
    Style::default().fg(Color::Magenta)
}

pub fn source() -> Style {
    Style::default().fg(Color::Blue)
}

pub fn error() -> Style {
    Style::default().fg(Color::Red)
}

pub fn ok() -> Style {
    Style::default().fg(Color::Green)
}

pub fn warn() -> Style {
    Style::default().fg(Color::Magenta)
}

pub fn added() -> Style {
    Style::default().fg(Color::Green)
}

pub fn removed() -> Style {
    Style::default().fg(Color::Red)
}

pub fn reversed() -> Style {
    Style::default().add_modifier(Modifier::REVERSED)
}
