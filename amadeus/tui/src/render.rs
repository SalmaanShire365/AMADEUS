//! Turns scrollback entries into styled, wrapped lines.
//!
//! Agent answers are markdown (pulldown-cmark) with any `<think>…</think>`
//! reasoning folded into a single dimmed line. Everything else is plain text
//! with a gutter marker. Wrapping is done here, span by span, so styles
//! survive line breaks and the cursor-free scrollback never needs ratatui's
//! own wrapping.

use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::scrollback::{Entry, Kind, View};
use crate::theme;

pub fn render_entry(e: &Entry, width: u16, view: View) -> Vec<Line<'static>> {
    let width = (width as usize).max(8);
    let mut out = match e.kind {
        Kind::Agent => render_agent(&e.text, width, view),
        Kind::User => plain(&e.text, width, "› ", theme::accent_bold(), theme::bold()),
        Kind::System => plain(&e.text, width, "· ", theme::dim(), theme::dim()),
        Kind::Error => plain(&e.text, width, "! ", theme::error(), theme::error()),
        Kind::Tool => render_tool(&e.text, width),
        Kind::Source => render_sources(&e.text, width),
    };
    if out.is_empty() {
        out.push(Line::default());
    }
    out.push(Line::default()); // breathing room between entries
    out
}

// ---------- think folding ----------

#[derive(Debug, PartialEq)]
pub enum Segment<'a> {
    Text(&'a str),
    Think { text: &'a str, closed: bool },
}

/// Split model output into visible text and reasoning blocks.
///
/// Handles a `</think>` with no opener (some chat templates put the opening
/// tag in the prompt, so only the close shows up in the output) and an
/// unclosed `<think>` (still streaming).
pub fn split_think(text: &str) -> Vec<Segment<'_>> {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let mut out = Vec::new();
    let mut rest = text;

    if let Some(c) = rest.find(CLOSE) {
        if rest.find(OPEN).is_none_or(|o| o > c) {
            out.push(Segment::Think {
                text: &rest[..c],
                closed: true,
            });
            rest = &rest[c + CLOSE.len()..];
        }
    }

    while !rest.is_empty() {
        match rest.find(OPEN) {
            None => {
                out.push(Segment::Text(rest));
                break;
            }
            Some(o) => {
                if o > 0 {
                    out.push(Segment::Text(&rest[..o]));
                }
                let body = &rest[o + OPEN.len()..];
                match body.find(CLOSE) {
                    Some(c) => {
                        out.push(Segment::Think {
                            text: &body[..c],
                            closed: true,
                        });
                        rest = &body[c + CLOSE.len()..];
                    }
                    None => {
                        out.push(Segment::Think {
                            text: body,
                            closed: false,
                        });
                        break;
                    }
                }
            }
        }
    }
    out
}

/// Answer text with reasoning removed, for history and run logs.
pub fn strip_think(text: &str) -> String {
    split_think(text)
        .into_iter()
        .filter_map(|s| match s {
            Segment::Text(t) => Some(t),
            Segment::Think { .. } => None,
        })
        .collect::<String>()
        .trim()
        .to_string()
}

fn render_agent(text: &str, width: usize, view: View) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    for seg in split_think(text) {
        match seg {
            Segment::Think { text, closed } => {
                let body = text.trim();
                if view.show_thinking && !body.is_empty() {
                    out.extend(plain(body, width, "┊ ", theme::dim(), theme::dim_italic()));
                } else if !closed {
                    out.push(Line::from(Span::styled("▸ thinking…", theme::dim_italic())));
                } else if !body.is_empty() {
                    let n = body.lines().count();
                    let noun = if n == 1 { "line" } else { "lines" };
                    out.push(Line::from(Span::styled(
                        format!("▸ thought for {n} {noun} · ctrl+o to show"),
                        theme::dim_italic(),
                    )));
                }
            }
            Segment::Text(t) => {
                let t = t.trim_start_matches(['\n', '\r']);
                if !t.trim().is_empty() {
                    out.extend(markdown(t, width));
                }
            }
        }
    }
    out
}

// ---------- tool and source entries ----------

/// First line is the call (`read_file notes/a.txt`), the rest is a short
/// result preview, dimmed.
fn render_tool(text: &str, width: usize) -> Vec<Line<'static>> {
    let (head, body) = text.split_once('\n').unwrap_or((text, ""));
    let mut out = plain(head, width, "» ", theme::tool(), theme::tool());
    if !body.trim().is_empty() {
        out.extend(plain(body, width, "  ", theme::dim(), theme::dim()));
    }
    out
}

fn render_sources(text: &str, width: usize) -> Vec<Line<'static>> {
    let (head, body) = text.split_once('\n').unwrap_or((text, ""));
    let mut out = plain(head, width, "  ", theme::dim(), theme::dim());
    for line in body.lines() {
        out.extend(plain(line, width, "    ", theme::source(), theme::source()));
    }
    out
}

/// Plain text with a gutter marker on the first line and matching
/// indentation on every other line.
pub fn plain(
    text: &str,
    width: usize,
    marker: &str,
    marker_style: Style,
    body: Style,
) -> Vec<Line<'static>> {
    let pad = " ".repeat(marker.width());
    let mut out = Vec::new();
    let mut first = true;
    for raw in text.split('\n') {
        let raw = raw.trim_end_matches('\r');
        if raw.is_empty() {
            out.push(Line::default());
            continue;
        }
        let raw = raw.replace('\t', "    ");
        let lead = if first { marker } else { pad.as_str() };
        first = false;
        out.extend(wrap_spans(
            &[(raw, body)],
            width,
            &[Span::styled(lead.to_string(), marker_style)],
            &[Span::raw(pad.clone())],
        ));
    }
    out
}

// ---------- span wrapping ----------

struct Wrapper {
    width: usize,
    rest: Vec<Span<'static>>,
    lines: Vec<Line<'static>>,
    cur: Vec<Span<'static>>,
    body_start: usize,
    used: usize,
    avail: usize,
    /// True until the first word lands, so leading indentation is kept on the
    /// first row and dropped on wrapped continuation rows.
    para_start: bool,
}

fn prefix_width(p: &[Span]) -> usize {
    p.iter().map(|s| s.content.width()).sum()
}

impl Wrapper {
    fn new(width: usize, first: &[Span<'static>], rest: &[Span<'static>]) -> Wrapper {
        Wrapper {
            width,
            rest: rest.to_vec(),
            lines: Vec::new(),
            cur: first.to_vec(),
            body_start: first.len(),
            used: 0,
            avail: width.saturating_sub(prefix_width(first)).max(1),
            para_start: true,
        }
    }

    fn append(&mut self, s: &str, style: Style) {
        if self.cur.len() > self.body_start {
            let last = self.cur.last_mut().unwrap();
            if last.style == style {
                last.content.to_mut().push_str(s);
                self.used += s.width();
                return;
            }
        }
        self.cur.push(Span::styled(s.to_string(), style));
        self.used += s.width();
    }

    fn break_line(&mut self) {
        // Drop trailing spaces so a wrapped row doesn't end in whitespace.
        while self.cur.len() > self.body_start {
            let last = self.cur.last_mut().unwrap();
            let trimmed = last.content.trim_end().len();
            if trimmed == 0 {
                self.cur.pop();
                continue;
            }
            last.content.to_mut().truncate(trimmed);
            break;
        }
        let row = std::mem::replace(&mut self.cur, self.rest.clone());
        self.lines.push(Line::from(row));
        self.body_start = self.cur.len();
        self.used = 0;
        self.avail = self.width.saturating_sub(prefix_width(&self.rest)).max(1);
    }

    fn word(&mut self, w: &str, style: Style) {
        self.para_start = false;
        let ww = w.width();
        if self.used > 0 && self.used + ww > self.avail {
            self.break_line();
        }
        if ww <= self.avail {
            self.append(w, style);
            return;
        }
        // A word longer than the row: hard-split it.
        let mut buf = String::new();
        for ch in w.chars() {
            let cw = ch.width().unwrap_or(0);
            if self.used + buf.width() + cw > self.avail && (self.used > 0 || !buf.is_empty()) {
                let piece = std::mem::take(&mut buf);
                self.append(&piece, style);
                self.break_line();
            }
            buf.push(ch);
        }
        if !buf.is_empty() {
            self.append(&buf, style);
        }
    }

    fn space(&mut self, s: &str, style: Style) {
        if self.used == 0 && !self.para_start {
            return;
        }
        if self.used + s.width() > self.avail {
            self.break_line();
            return;
        }
        self.append(s, style);
    }

    fn finish(mut self) -> Vec<Line<'static>> {
        if self.cur.len() > self.body_start || self.lines.is_empty() {
            self.break_line();
        }
        self.lines
    }
}

/// Greedy word wrap over styled spans.
pub fn wrap_spans(
    spans: &[(String, Style)],
    width: usize,
    first: &[Span<'static>],
    rest: &[Span<'static>],
) -> Vec<Line<'static>> {
    let mut w = Wrapper::new(width, first, rest);
    for (text, style) in spans {
        let mut start = 0;
        let mut in_space: Option<bool> = None;
        for (i, ch) in text.char_indices() {
            let sp = ch == ' ';
            match in_space {
                Some(prev) if prev != sp => {
                    let piece = &text[start..i];
                    if prev {
                        w.space(piece, *style);
                    } else {
                        w.word(piece, *style);
                    }
                    start = i;
                }
                _ => {}
            }
            in_space = Some(sp);
        }
        if let Some(sp) = in_space {
            let piece = &text[start..];
            if sp {
                w.space(piece, *style);
            } else {
                w.word(piece, *style);
            }
        }
    }
    w.finish()
}

// ---------- markdown ----------

struct ListState {
    next: Option<u64>,
    marker_w: usize,
}

struct Table {
    rows: Vec<Vec<String>>,
    row: Vec<String>,
    cell: String,
    has_head: bool,
}

struct Md {
    width: usize,
    out: Vec<Line<'static>>,
    spans: Vec<(String, Style)>,
    styles: Vec<Style>,
    lists: Vec<ListState>,
    bullet: Option<String>,
    quote: usize,
    code: Option<String>,
    table: Option<Table>,
}

pub fn markdown(text: &str, width: usize) -> Vec<Line<'static>> {
    let mut md = Md {
        width,
        out: Vec::new(),
        spans: Vec::new(),
        styles: Vec::new(),
        lists: Vec::new(),
        bullet: None,
        quote: 0,
        code: None,
        table: None,
    };
    let opts = Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TASKLISTS;
    for ev in Parser::new_ext(text, opts) {
        md.event(ev);
    }
    md.flush();
    if let Some(buf) = md.code.take() {
        md.emit_code(&buf); // unclosed fence while streaming
    }
    while md.out.last().is_some_and(|l| l.width() == 0) {
        md.out.pop();
    }
    md.out
}

impl Md {
    fn style(&self) -> Style {
        self.styles
            .iter()
            .fold(Style::default(), |acc, s| acc.patch(*s))
    }

    fn blank(&mut self) {
        if self.lists.is_empty() && self.out.last().is_some_and(|l| l.width() > 0) {
            self.out.push(Line::default());
        }
    }

    fn quote_prefix(&self) -> String {
        "▎ ".repeat(self.quote)
    }

    /// (first-row prefix, continuation prefix) for the current block.
    fn prefixes(&mut self) -> (Vec<Span<'static>>, Vec<Span<'static>>) {
        let q = self.quote_prefix();
        let (first, rest) = match self.lists.split_last() {
            None => (String::new(), String::new()),
            Some((inner, outer)) => {
                let base: usize = outer.iter().map(|l| l.marker_w).sum();
                let cont = " ".repeat(base + inner.marker_w);
                let first = match self.bullet.take() {
                    Some(b) => format!("{}{b}", " ".repeat(base)),
                    None => cont.clone(),
                };
                (first, cont)
            }
        };
        let qs = theme::dim();
        (
            vec![
                Span::styled(q.clone(), qs),
                Span::styled(first, theme::accent()),
            ],
            vec![Span::styled(q, qs), Span::raw(rest)],
        )
    }

    fn flush(&mut self) {
        if self.spans.is_empty() {
            return;
        }
        let spans = std::mem::take(&mut self.spans);
        let (first, rest) = self.prefixes();
        self.out
            .extend(wrap_spans(&spans, self.width, &first, &rest));
    }

    fn text(&mut self, t: &str, extra: Style) {
        if let Some(buf) = self.code.as_mut() {
            buf.push_str(t);
        } else if let Some(tb) = self.table.as_mut() {
            tb.cell.push_str(t);
        } else {
            let st = self.style().patch(extra);
            self.spans.push((t.to_string(), st));
        }
    }

    fn event(&mut self, ev: Event) {
        match ev {
            Event::Start(tag) => self.start(tag),
            Event::End(tag) => self.end(tag),
            Event::Text(t) => self.text(&t, Style::default()),
            Event::Code(t) => self.text(&t, theme::code()),
            Event::Html(t) | Event::InlineHtml(t) => {
                self.text(t.trim_end_matches('\n'), Style::default())
            }
            Event::InlineMath(t) | Event::DisplayMath(t) => self.text(&t, theme::code()),
            Event::FootnoteReference(t) => self.text(&format!("[{t}]"), Style::default()),
            Event::SoftBreak => self.text(" ", Style::default()),
            Event::HardBreak => self.flush(),
            Event::Rule => {
                self.flush();
                let n = self.width.min(40);
                self.out
                    .push(Line::from(Span::styled("─".repeat(n), theme::dim())));
                self.blank();
            }
            Event::TaskListMarker(done) => {
                self.text(if done { "[x] " } else { "[ ] " }, Style::default())
            }
        }
    }

    fn start(&mut self, tag: Tag) {
        match tag {
            Tag::Heading { .. } => {
                self.flush();
                self.styles.push(theme::accent_bold());
            }
            Tag::BlockQuote(_) => {
                self.flush();
                self.quote += 1;
            }
            Tag::CodeBlock(_) => {
                self.flush();
                self.code = Some(String::new());
            }
            Tag::List(start) => {
                self.flush();
                self.lists.push(ListState {
                    next: start,
                    marker_w: 2,
                });
            }
            Tag::Item => {
                self.flush();
                if let Some(l) = self.lists.last_mut() {
                    let marker = match l.next {
                        Some(n) => {
                            l.next = Some(n + 1);
                            format!("{n}. ")
                        }
                        None => "• ".to_string(),
                    };
                    l.marker_w = marker.width();
                    self.bullet = Some(marker);
                }
            }
            Tag::Emphasis => self
                .styles
                .push(Style::default().add_modifier(Modifier::ITALIC)),
            Tag::Strong => self.styles.push(theme::bold()),
            Tag::Strikethrough => self
                .styles
                .push(Style::default().add_modifier(Modifier::CROSSED_OUT)),
            Tag::Link { .. } => self
                .styles
                .push(theme::accent().add_modifier(Modifier::UNDERLINED)),
            Tag::Image { .. } => self.styles.push(theme::dim()),
            Tag::Table(_) => {
                self.flush();
                self.table = Some(Table {
                    rows: Vec::new(),
                    row: Vec::new(),
                    cell: String::new(),
                    has_head: false,
                });
            }
            Tag::TableHead | Tag::TableRow => {
                if let Some(t) = self.table.as_mut() {
                    t.row.clear();
                }
            }
            Tag::TableCell => {
                if let Some(t) = self.table.as_mut() {
                    t.cell.clear();
                }
            }
            _ => {}
        }
    }

    fn end(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph => {
                self.flush();
                self.blank();
            }
            TagEnd::Heading(_) => {
                self.flush();
                self.styles.pop();
                self.blank();
            }
            TagEnd::BlockQuote(_) => {
                self.flush();
                self.quote = self.quote.saturating_sub(1);
                self.blank();
            }
            TagEnd::CodeBlock => {
                if let Some(buf) = self.code.take() {
                    self.emit_code(&buf);
                }
                self.blank();
            }
            TagEnd::List(_) => {
                self.flush();
                self.lists.pop();
                self.blank();
            }
            TagEnd::Item => {
                self.flush();
                self.bullet = None;
            }
            TagEnd::Emphasis
            | TagEnd::Strong
            | TagEnd::Strikethrough
            | TagEnd::Link
            | TagEnd::Image => {
                self.styles.pop();
            }
            TagEnd::HtmlBlock => self.flush(),
            TagEnd::TableCell => {
                if let Some(t) = self.table.as_mut() {
                    let c = std::mem::take(&mut t.cell);
                    t.row.push(c.trim().to_string());
                }
            }
            TagEnd::TableHead => {
                if let Some(t) = self.table.as_mut() {
                    let r = std::mem::take(&mut t.row);
                    t.rows.push(r);
                    t.has_head = true;
                }
            }
            TagEnd::TableRow => {
                if let Some(t) = self.table.as_mut() {
                    let r = std::mem::take(&mut t.row);
                    t.rows.push(r);
                }
            }
            TagEnd::Table => {
                if let Some(t) = self.table.take() {
                    self.emit_table(t);
                }
                self.blank();
            }
            _ => {}
        }
    }

    fn indent(&self) -> String {
        let n: usize = self.lists.iter().map(|l| l.marker_w).sum();
        format!("{}{}", self.quote_prefix(), " ".repeat(n))
    }

    fn emit_code(&mut self, buf: &str) {
        let indent = self.indent();
        let bar = "│ ";
        let avail = self
            .width
            .saturating_sub(indent.width() + bar.width())
            .max(4);
        for raw in buf.trim_end_matches('\n').split('\n') {
            let raw = raw.replace('\t', "    ");
            let mut rows = vec![String::new()];
            let mut w = 0;
            for ch in raw.chars() {
                let cw = ch.width().unwrap_or(0);
                if w + cw > avail {
                    rows.push(String::new());
                    w = 0;
                }
                rows.last_mut().unwrap().push(ch);
                w += cw;
            }
            for r in rows {
                self.out.push(Line::from(vec![
                    Span::raw(indent.clone()),
                    Span::styled(bar, theme::dim()),
                    Span::styled(r, theme::code()),
                ]));
            }
        }
    }

    fn emit_table(&mut self, t: Table) {
        let indent = self.indent();
        let ncols = t.rows.iter().map(Vec::len).max().unwrap_or(0);
        if ncols == 0 {
            return;
        }
        let mut widths = vec![0usize; ncols];
        for r in &t.rows {
            for (i, c) in r.iter().enumerate() {
                widths[i] = widths[i].max(c.width());
            }
        }
        let sep = " │ ";
        let total: usize = widths.iter().sum::<usize>() + sep.width() * (ncols - 1);
        let avail = self.width.saturating_sub(indent.width());

        if total <= avail {
            for (ri, r) in t.rows.iter().enumerate() {
                let head = t.has_head && ri == 0;
                let mut spans = vec![Span::raw(indent.clone())];
                for (i, w) in widths.iter().enumerate() {
                    let c = r.get(i).map(String::as_str).unwrap_or("");
                    let padded = format!("{c}{}", " ".repeat(w.saturating_sub(c.width())));
                    let st = if head {
                        theme::bold()
                    } else {
                        Style::default()
                    };
                    spans.push(Span::styled(padded, st));
                    if i + 1 < ncols {
                        spans.push(Span::styled(sep, theme::dim()));
                    }
                }
                self.out.push(Line::from(spans));
                if head {
                    let rule = widths
                        .iter()
                        .map(|w| "─".repeat(*w))
                        .collect::<Vec<_>>()
                        .join("─┼─");
                    self.out.push(Line::from(vec![
                        Span::raw(indent.clone()),
                        Span::styled(rule, theme::dim()),
                    ]));
                }
            }
        } else {
            // Too wide to align: one wrapped row per table row.
            for (ri, r) in t.rows.iter().enumerate() {
                let st = if t.has_head && ri == 0 {
                    theme::bold()
                } else {
                    Style::default()
                };
                let joined = r.join(" │ ");
                let pre = vec![Span::raw(indent.clone())];
                self.out
                    .extend(wrap_spans(&[(joined, st)], self.width, &pre, &pre));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_of(lines: &[Line]) -> Vec<String> {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect()
    }

    #[test]
    fn think_block_is_split_from_the_answer() {
        let s = split_think("<think>hmm\nok</think>\n\nHello");
        assert_eq!(
            s,
            vec![
                Segment::Think {
                    text: "hmm\nok",
                    closed: true
                },
                Segment::Text("\n\nHello"),
            ]
        );
    }

    #[test]
    fn unclosed_think_is_still_streaming() {
        assert_eq!(
            split_think("<think>still going"),
            vec![Segment::Think {
                text: "still going",
                closed: false
            }]
        );
    }

    #[test]
    fn close_without_open_folds_the_prefix() {
        assert_eq!(
            split_think("reasoning</think>answer"),
            vec![
                Segment::Think {
                    text: "reasoning",
                    closed: true
                },
                Segment::Text("answer"),
            ]
        );
        assert_eq!(strip_think("<think>x</think> answer "), "answer");
    }

    #[test]
    fn folded_thinking_renders_as_one_dim_line() {
        let lines = render_agent("<think>a\nb\nc</think>Done.", 40, View::default());
        let t = text_of(&lines);
        assert_eq!(t[0], "▸ thought for 3 lines · ctrl+o to show");
        assert_eq!(t[1], "Done.");
        assert!(!t.iter().any(|l| l.contains("<think>")));
    }

    #[test]
    fn expanded_thinking_shows_the_reasoning() {
        let view = View {
            show_thinking: true,
        };
        let t = text_of(&render_agent("<think>because</think>ok", 40, view));
        assert_eq!(t[0], "┊ because");
    }

    #[test]
    fn wrap_keeps_words_whole_and_indents_continuations() {
        let t = text_of(&plain(
            "aaa bbb ccc",
            9,
            "› ",
            Style::default(),
            Style::default(),
        ));
        assert_eq!(t, vec!["› aaa bbb", "  ccc"]);
    }

    #[test]
    fn long_words_are_hard_split() {
        let t = text_of(&plain(
            "abcdefghij",
            6,
            "",
            Style::default(),
            Style::default(),
        ));
        assert_eq!(t, vec!["abcdef", "ghij"]);
    }

    #[test]
    fn first_row_keeps_leading_indentation() {
        let t = text_of(&plain(
            "  indented",
            40,
            "",
            Style::default(),
            Style::default(),
        ));
        assert_eq!(t, vec!["  indented"]);
    }

    #[test]
    fn markdown_lists_and_code_blocks() {
        let md = "# Title\n\n- one\n- two\n\n```\nlet x = 1;\n```\n";
        let t = text_of(&markdown(md, 40));
        assert_eq!(t, vec!["Title", "", "• one", "• two", "", "│ let x = 1;"]);
    }

    #[test]
    fn numbered_list_continuation_aligns_under_text() {
        let t = text_of(&markdown("1. alpha beta gamma", 12));
        assert_eq!(t, vec!["1. alpha", "   beta", "   gamma"]);
    }

    #[test]
    fn tables_align_when_they_fit() {
        let t = text_of(&markdown("| a | bb |\n|---|---|\n| ccc | d |\n", 40));
        assert_eq!(t, vec!["a   │ bb", "────┼───", "ccc │ d "]);
    }

    #[test]
    fn inline_styles_do_not_leak_markers() {
        let t = text_of(&markdown("use **bold** and `code`", 40));
        assert_eq!(t, vec!["use bold and code"]);
    }
}
