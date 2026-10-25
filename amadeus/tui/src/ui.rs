use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::app::{layout_input, App};
use crate::health::Status;
use crate::hint::Hint;
use crate::theme;

pub fn draw(f: &mut Frame, app: &mut App) {
    let area = f.area();

    // Input box grows with content, up to 8 rows.
    let inner_w = area.width.saturating_sub(4).max(4) as usize;
    let (rows, crow, ccol) = layout_input(&app.input, app.cursor, inner_w);
    let input_h = (rows.len().clamp(1, 8) + 2) as u16;

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(input_h),
            Constraint::Length(1),
        ])
        .split(area);

    draw_header(f, app, chunks[0]);
    draw_scrollback(f, app, chunks[1]);
    draw_input(f, app, chunks[2], &rows, crow, ccol);
    draw_status(f, app, chunks[3]);

    if app.pending.is_some() {
        draw_approval(f, app, chunks[1]);
    } else if app.picker.is_some() {
        draw_picker(f, app, chunks[1]);
    } else if let Hint::Menu(cmds) = app.hint() {
        // Completion popup sits directly above the input, overlapping scrollback.
        draw_menu(f, &cmds, chunks[2]);
    }
    if app.show_help {
        draw_help(f, area);
    }
}

/// `~/demo` rather than `/home/me/demo`, cut from the left to fit.
fn short_path(p: &std::path::Path, max: usize) -> String {
    let s = p.display().to_string();
    let s = match std::env::var("HOME") {
        Ok(h) if !h.is_empty() && s.starts_with(&h) => format!("~{}", &s[h.len()..]),
        _ => s,
    };
    let n = s.chars().count();
    if n <= max {
        s
    } else {
        format!("…{}", s.chars().skip(n + 1 - max).collect::<String>())
    }
}

/// One line: what's running, on which model, where, and whether Ollama is up.
fn draw_header(f: &mut Frame, app: &App, area: Rect) {
    let (dot, dot_style, word) = match app.ollama_status() {
        Some(Status::Ok) => ("●", theme::ok(), "ollama"),
        Some(Status::Degraded) => ("●", theme::warn(), "model missing"),
        Some(Status::Down) => ("●", theme::error(), "ollama down"),
        None => ("○", theme::dim(), "checking"),
    };
    let template = app
        .template
        .as_ref()
        .map_or("chat".to_string(), |t| t.title.clone());
    let model = format!("{} {}", app.tier.label(), app.cfg.model(app.tier));
    let sep = || Span::styled("  │  ", theme::dim());
    let mut spans = vec![
        Span::styled(" AMADEUS", theme::accent_bold()),
        sep(),
        Span::styled(template, theme::bold()),
        sep(),
        Span::raw(model),
        sep(),
    ];
    let status = [
        sep(),
        Span::styled(dot, dot_style),
        Span::styled(format!(" {word}"), theme::dim()),
    ];
    // The workspace path gets whatever room is left, and shrinks first.
    let used: usize = spans.iter().chain(status.iter()).map(|s| s.width()).sum();
    let room = (area.width as usize).saturating_sub(used);
    if room >= 6 {
        spans.push(Span::styled(short_path(&app.cfg.root, room), theme::dim()));
    } else {
        spans.pop();
    }
    spans.extend(status);
    let line = Line::from(spans);
    f.render_widget(Paragraph::new(line), area);
}

/// A box anchored to the bottom of `area`, full width minus a margin.
fn bottom_box(area: Rect, height: u16) -> Rect {
    let h = height.min(area.height);
    Rect {
        x: area.x + 1,
        y: area.y + area.height - h,
        width: area.width.saturating_sub(2),
        height: h,
    }
}

fn diff_line(l: &str) -> Line<'static> {
    let st = if l.starts_with("+++") || l.starts_with("---") || l.starts_with("@@") {
        theme::dim()
    } else if l.starts_with('+') {
        theme::added()
    } else if l.starts_with('-') {
        theme::removed()
    } else if l.starts_with('$') {
        theme::accent_bold()
    } else {
        Style::default()
    };
    Line::from(Span::styled(l.replace('\t', "    "), st))
}

fn draw_approval(f: &mut Frame, app: &mut App, area: Rect) {
    let Some(p) = app.pending.as_mut() else {
        return;
    };
    let lines: Vec<&str> = p.body.lines().collect();
    let max_h = (area.height as usize * 3 / 4).max(6);
    let room = max_h.saturating_sub(5).max(1);
    let max_scroll = lines.len().saturating_sub(room);
    p.scroll = p.scroll.min(max_scroll);

    let mut body: Vec<Line> = vec![
        Line::from(Span::styled(p.title.clone(), theme::bold())),
        Line::default(),
    ];
    body.extend(lines.iter().skip(p.scroll).take(room).map(|l| diff_line(l)));
    let hidden = lines.len().saturating_sub(p.scroll + room);
    if hidden > 0 || p.scroll > 0 {
        body.push(Line::from(Span::styled(
            format!("… {hidden} more lines · ↑/↓ scroll"),
            theme::dim_italic(),
        )));
    }

    let h = (body.len() + 2) as u16;
    let rect = bottom_box(area, h);
    let keys = Line::from(vec![
        Span::styled(" y ", theme::reversed()),
        Span::raw(" approve  "),
        Span::styled(" n ", theme::reversed()),
        Span::raw(" skip  "),
        Span::styled(" a ", theme::reversed()),
        Span::raw(" approve all for this task  "),
        Span::styled("ctrl+c stop ", theme::dim()),
    ]);
    f.render_widget(Clear, rect);
    f.render_widget(
        Paragraph::new(body).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme::warn())
                .title(Span::styled(
                    " needs your approval ",
                    theme::warn().add_modifier(ratatui::style::Modifier::BOLD),
                ))
                .title_bottom(keys),
        ),
        rect,
    );
}

fn draw_picker(f: &mut Frame, app: &App, area: Rect) {
    let Some(sel) = app.picker else { return };
    let mut lines: Vec<Line> = Vec::new();
    for (i, t) in app.templates.iter().enumerate() {
        let on = i == sel;
        let marker = if on { "▸ " } else { "  " };
        let st = if on {
            theme::accent_bold()
        } else {
            theme::bold()
        };
        lines.push(Line::from(vec![
            Span::styled(format!("{marker}{}. ", i + 1), st),
            Span::styled(t.title.clone(), st),
            Span::styled(format!("  /use {}", t.name), theme::dim()),
        ]));
        lines.push(Line::from(Span::styled(
            format!("     {}", t.description),
            theme::dim(),
        )));
    }
    let rect = bottom_box(area, (lines.len() + 2) as u16);
    f.render_widget(Clear, rect);
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme::accent())
                .title(" choose a template ")
                .title_bottom(Span::styled(
                    " ↑↓ move · enter or 1-9 choose · esc close ",
                    theme::dim(),
                )),
        ),
        rect,
    );
}

const HELP: &[(&str, &str)] = &[
    ("", "TASKS"),
    (
        "/new",
        "pick a template (folder summary, spreadsheet, replies, code)",
    ),
    (
        "/use NAME",
        "switch to a template by name; its example fills the input",
    ),
    (
        "/auto",
        "auto-approve writes and commands for the next task only",
    ),
    ("/reset", "start a fresh conversation, same template"),
    ("/chat", "leave the template: plain chat, no tools"),
    ("/log", "where this task's audit log is"),
    ("", "APPROVALS"),
    (
        "y / n / a",
        "approve · skip · approve the rest of this task",
    ),
    ("↑ ↓", "scroll a long diff"),
    ("", "KEYS"),
    ("enter", "send · alt+enter for a newline"),
    ("↑ ↓", "previous inputs"),
    (
        "ctrl+c",
        "stop the current task (twice on empty input quits)",
    ),
    ("ctrl+t", "cycle the model: fast / main / heavy"),
    ("ctrl+o", "show or hide the model's reasoning"),
    ("pgup pgdn, wheel", "scroll · end jumps back to the bottom"),
    ("ctrl+l", "clear the screen"),
    ("", "CODE"),
    ("/rag Q", "answer from the indexed codebase, with sources"),
    (
        "/index DIR",
        "build the code index · /debug shows retrieval distances",
    ),
    ("/shell CMD", "run a command yourself"),
];

fn draw_help(f: &mut Frame, area: Rect) {
    let mut lines = Vec::new();
    for (k, v) in HELP {
        if k.is_empty() {
            if !lines.is_empty() {
                lines.push(Line::default());
            }
            lines.push(Line::from(Span::styled(
                v.to_string(),
                theme::accent_bold(),
            )));
        } else {
            lines.push(Line::from(vec![
                Span::styled(format!("  {k:<18}"), theme::bold()),
                Span::raw(v.to_string()),
            ]));
        }
    }
    let w = area.width.min(84);
    let h = ((lines.len() + 2) as u16).min(area.height);
    let rect = Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, rect);
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme::accent())
                .title(" help ")
                .title_bottom(Span::styled(" any key closes ", theme::dim())),
        ),
        rect,
    );
}

fn draw_scrollback(f: &mut Frame, app: &mut App, area: Rect) {
    let h = area.height as usize;
    let (pinned, scroll) = (app.pinned, app.scroll);
    let (stable, tail) = app.sb.lines(area.width);
    let total = stable.len() + tail.len();
    let max_off = total.saturating_sub(h);

    let start = if pinned { max_off } else { scroll.min(max_off) };
    let end = (start + h).min(total);
    let visible: Vec<Line> = (start..end)
        .map(|i| {
            if i < stable.len() {
                stable[i].clone()
            } else {
                tail[i - stable.len()].clone()
            }
        })
        .collect();
    f.render_widget(Paragraph::new(visible), area);

    app.scroll = start;
    if !pinned && scroll >= max_off {
        app.pinned = true;
    }
}

fn draw_input(f: &mut Frame, app: &App, area: Rect, rows: &[String], crow: usize, ccol: usize) {
    let title = if app.busy.is_some() {
        " working "
    } else {
        " task "
    };
    let border = if app.busy.is_some() {
        theme::dim()
    } else {
        theme::accent()
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border)
        .title(title);

    let visible_rows = rows.len().min(8);
    let skip = rows.len() - visible_rows;
    let body: Vec<Line> = rows[skip..].iter().map(|r| Line::from(r.clone())).collect();

    f.render_widget(Paragraph::new(body).block(block), area);

    let cy = area.y + 1 + crow.saturating_sub(skip) as u16;
    let cx = area.x + 1 + ccol as u16;
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
        Paragraph::new(Line::from(Span::styled(left, theme::dim_italic()))),
        cols[0],
    );
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(right, theme::dim()))),
        cols[1],
    );
}

fn draw_menu(f: &mut Frame, cmds: &[&crate::hint::Cmd], input_area: Rect) {
    let n = cmds.len().min(8) as u16;
    if n == 0 || input_area.y < n + 2 {
        return;
    }
    let width = input_area.width.min(64);
    let area = Rect {
        x: input_area.x,
        y: input_area.y - (n + 2),
        width,
        height: n + 2,
    };

    let lines: Vec<Line> = cmds
        .iter()
        .take(8)
        .map(|c| {
            Line::from(vec![
                Span::styled(format!("{:<10}", c.name), theme::accent_bold()),
                Span::styled(format!("{:<12}", c.args), Style::default()),
                Span::styled(c.help.to_string(), theme::dim()),
            ])
        })
        .collect();

    f.render_widget(Clear, area);
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme::dim())
                .title(" tab to complete "),
        ),
        area,
    );
}
