use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::app::{layout_input, App};
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
