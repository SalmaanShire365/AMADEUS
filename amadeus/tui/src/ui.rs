use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::app::{layout_input, App, Entry, Kind};
use crate::hint::Hint;

fn kind_style(kind: Kind) -> (&'static str, Style) {
    match kind {
        Kind::User => (
            "› ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Kind::Agent => ("", Style::default()),
        Kind::System => ("· ", Style::default().fg(Color::DarkGray)),
        Kind::Source => ("  ", Style::default().fg(Color::Blue)),
        Kind::Error => ("! ", Style::default().fg(Color::Red)),
    }
}

/// Word-wrap one entry into styled lines, with a gutter marker on the first
/// line and matching indentation on continuations.
fn wrap_entry(entry: &Entry, width: u16) -> Vec<Line<'static>> {
    let (prefix, style) = kind_style(entry.kind);
    let total = width.max(8) as usize;
    let inner = total.saturating_sub(prefix.len()).max(4);
    let pad = " ".repeat(prefix.len());

    let mut out: Vec<Line<'static>> = Vec::new();
    let mut first = true;
    for raw in entry.text.split('\n') {
        if raw.is_empty() {
            out.push(Line::from(""));
            continue;
        }
        for piece in textwrap::wrap(raw, inner) {
            let gutter = if first {
                prefix.to_string()
            } else {
                pad.clone()
            };
            first = false;
            out.push(Line::from(vec![
                Span::styled(gutter, style),
                Span::styled(piece.to_string(), style),
            ]));
        }
    }
    if out.is_empty() {
        out.push(Line::from(""));
    }
    out.push(Line::from("")); // breathing room between entries
    out
}

/// Wrap everything except the final entry once, and re-wrap only the final
/// entry each frame. That keeps a streaming answer cheap no matter how long
/// the session's scrollback gets.
fn ensure_wrapped(app: &mut App, width: u16) -> Vec<Line<'static>> {
    if app.wrap_width != width {
        app.wrap_width = width;
        app.invalidate_wrap();
    }
    while app.entries.len() > 0 && app.stable_upto + 1 < app.entries.len() {
        let lines = wrap_entry(&app.entries[app.stable_upto], width);
        app.stable.extend(lines);
        app.stable_upto += 1;
    }
    match app.entries.last() {
        Some(e) if app.stable_upto < app.entries.len() => wrap_entry(e, width),
        _ => Vec::new(),
    }
}

pub fn draw(f: &mut Frame, app: &mut App) {
    let area = f.area();

    // Input box grows with content, up to 8 rows.
    let inner_w = area.width.saturating_sub(4).max(4) as usize;
    let (rows, crow, ccol) = layout_input(&app.input, app.cursor, inner_w);
    let input_h = (rows.len().clamp(1, 8) + 2) as u16;

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),
            Constraint::Length(input_h),
            Constraint::Length(1),
        ])
        .split(area);

    draw_scrollback(f, app, chunks[0]);
    draw_input(f, app, chunks[1], &rows, crow, ccol);
    draw_status(f, app, chunks[2]);

    // Completion popup sits directly above the input, overlapping scrollback.
    if let Hint::Menu(cmds) = app.hint() {
        draw_menu(f, &cmds, chunks[1]);
    }
}

fn draw_scrollback(f: &mut Frame, app: &mut App, area: Rect) {
    let tail = ensure_wrapped(app, area.width);
    let total = app.stable.len() + tail.len();
    let h = area.height as usize;
    let max_off = total.saturating_sub(h);

    if app.pinned {
        app.scroll = max_off;
    } else {
        if app.scroll >= max_off {
            app.scroll = max_off;
            app.pinned = true;
        }
    }

    let start = app.scroll.min(max_off);
    let end = (start + h).min(total);
    let visible: Vec<Line> = (start..end)
        .map(|i| {
            if i < app.stable.len() {
                app.stable[i].clone()
            } else {
                tail[i - app.stable.len()].clone()
            }
        })
        .collect();

    f.render_widget(Paragraph::new(visible), area);
}

fn draw_input(f: &mut Frame, app: &App, area: Rect, rows: &[String], crow: usize, ccol: usize) {
    let title = if app.busy.is_some() {
        " working "
    } else {
        " task "
    };
    let border = if app.busy.is_some() {
        Color::DarkGray
    } else {
        Color::Cyan
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border))
        .title(title);

    let visible_rows = rows.len().min(8);
    let skip = rows.len() - visible_rows;
    let body: Vec<Line> = rows[skip..].iter().map(|r| Line::from(r.clone())).collect();

    f.render_widget(Paragraph::new(body).block(block), area);

    let cy = area.y + 1 + crow.saturating_sub(skip) as u16;
    let cx = area.x + 2 + ccol as u16;
    if cy < area.y + area.height - 1 {
        f.set_cursor_position((cx, cy));
    }
}

fn draw_status(f: &mut Frame, app: &App, area: Rect) {
    let right = app.status_right();
    let rw = (right.chars().count() as u16 + 1).min(area.width);
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(0), Constraint::Length(rw)])
        .split(area);

    let left = match app.hint() {
        Hint::Tip(t) => t,
        _ => String::new(),
    };
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            left,
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        ))),
        cols[0],
    );
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            right,
            Style::default().fg(Color::DarkGray),
        ))),
        cols[1],
    );
}

fn draw_menu(f: &mut Frame, cmds: &[&crate::hint::Cmd], input_area: Rect) {
    let n = cmds.len().min(6) as u16;
    if n == 0 || input_area.y < n + 2 {
        return;
    }
    let width = input_area.width.min(60);
    let area = Rect {
        x: input_area.x,
        y: input_area.y - (n + 2),
        width,
        height: n + 2,
    };

    let lines: Vec<Line> = cmds
        .iter()
        .take(6)
        .map(|c| {
            Line::from(vec![
                Span::styled(
                    format!("{:<10}", c.name),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(format!("{:<12}", c.args), Style::default().fg(Color::Gray)),
                Span::styled(c.help.to_string(), Style::default().fg(Color::DarkGray)),
            ])
        })
        .collect();

    f.render_widget(Clear, area);
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(" tab to complete "),
        ),
        area,
    );
}
