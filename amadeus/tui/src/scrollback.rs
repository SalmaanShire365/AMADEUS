//! The transcript: entries, the one entry currently being streamed into, and
//! a cache of wrapped lines.
//!
//! The streamed ("live") entry isn't necessarily the last one. A notice such
//! as "cancelling…" can land after it while tokens are still arriving, so
//! tokens go to the live index, never to `last_mut()`. The cache covers every
//! entry before the live one; the live entry and anything after it are
//! re-wrapped each frame (cheap: that tail is a handful of entries).

use ratatui::text::Line;

use crate::render;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    User,
    Agent,
    System,
    Source,
    Tool,
    Error,
}

#[derive(Debug)]
pub struct Entry {
    pub kind: Kind,
    pub text: String,
}

/// Render options that change how existing entries look. Changing any of
/// them invalidates the cache.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct View {
    pub show_thinking: bool,
}

#[derive(Default)]
pub struct Scrollback {
    entries: Vec<Entry>,
    live: Option<usize>,
    cache: Vec<Line<'static>>,
    cached_upto: usize,
    width: u16,
    view: View,
}

impl Scrollback {
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn push(&mut self, kind: Kind, text: String) {
        self.entries.push(Entry { kind, text });
    }

    /// Start a new live entry. Any previous live entry is closed.
    pub fn open_live(&mut self, kind: Kind) {
        self.entries.push(Entry {
            kind,
            text: String::new(),
        });
        self.live = Some(self.entries.len() - 1);
    }

    pub fn has_live(&self) -> bool {
        self.live.is_some()
    }

    /// Append to the live entry. Returns false (and drops the text) when
    /// nothing is live.
    pub fn append_live(&mut self, delta: &str) -> bool {
        match self.live.and_then(|i| self.entries.get_mut(i)) {
            Some(e) => {
                e.text.push_str(delta);
                true
            }
            None => false,
        }
    }

    pub fn close_live(&mut self) {
        // An empty live entry (cancelled before the first token) is noise.
        if let Some(i) = self.live.take() {
            if self.entries.get(i).is_some_and(|e| e.text.is_empty()) {
                // Safe for the cache: it only ever covers entries before `i`.
                self.entries.remove(i);
            }
        }
    }

    /// Clear the transcript but keep the entry that's being streamed into, so
    /// ctrl+l mid-answer doesn't orphan the rest of the answer.
    pub fn clear(&mut self) {
        let keep = self.live.map(|i| self.entries.swap_remove(i));
        self.entries.clear();
        self.live = None;
        if let Some(e) = keep {
            self.entries.push(e);
            self.live = Some(0);
        }
        self.invalidate();
    }

    pub fn view(&self) -> View {
        self.view
    }

    pub fn set_view(&mut self, v: View) {
        if v != self.view {
            self.view = v;
            self.invalidate();
        }
    }

    pub fn invalidate(&mut self) {
        self.cache.clear();
        self.cached_upto = 0;
    }

    /// Wrapped lines at `width`: (cached prefix, freshly rendered tail).
    pub fn lines(&mut self, width: u16) -> (&[Line<'static>], Vec<Line<'static>>) {
        if width != self.width {
            self.width = width;
            self.invalidate();
        }
        let frozen = self.live.unwrap_or(self.entries.len());
        if self.cached_upto > frozen {
            self.invalidate();
        }
        while self.cached_upto < frozen {
            let e = &self.entries[self.cached_upto];
            self.cache.extend(render::render_entry(e, width, self.view));
            self.cached_upto += 1;
        }
        let tail = self.entries[frozen..]
            .iter()
            .flat_map(|e| render::render_entry(e, width, self.view))
            .collect();
        (&self.cache, tail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(sb: &mut Scrollback, width: u16) -> Vec<String> {
        let (a, b) = sb.lines(width);
        a.iter()
            .chain(b.iter())
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .filter(|s: &String| !s.is_empty())
            .collect()
    }

    #[test]
    fn tokens_go_to_the_live_entry_not_the_last() {
        let mut sb = Scrollback::default();
        sb.open_live(Kind::Agent);
        sb.append_live("Hel");
        sb.push(Kind::System, "cancelling…".into());
        sb.append_live("lo");
        assert_eq!(sb.entries()[0].text, "Hello");
        assert_eq!(sb.entries()[1].text, "cancelling…");
    }

    #[test]
    fn cache_reflects_live_text_that_is_not_last() {
        let mut sb = Scrollback::default();
        sb.push(Kind::System, "hi".into());
        sb.open_live(Kind::Agent);
        sb.append_live("one");
        sb.push(Kind::System, "note".into());
        assert_eq!(flat(&mut sb, 40), vec!["· hi", "one", "· note"]);

        sb.append_live(" two");
        assert_eq!(flat(&mut sb, 40), vec!["· hi", "one two", "· note"]);

        // Once closed, the entry is frozen into the cache, and it's still right.
        sb.close_live();
        assert_eq!(flat(&mut sb, 40), vec!["· hi", "one two", "· note"]);
        assert_eq!(sb.cached_upto, 3);
    }

    #[test]
    fn append_without_live_is_dropped() {
        let mut sb = Scrollback::default();
        sb.push(Kind::System, "x".into());
        assert!(!sb.append_live("y"));
        assert_eq!(sb.entries()[0].text, "x");
    }

    #[test]
    fn clear_mid_stream_keeps_the_live_entry_writable() {
        let mut sb = Scrollback::default();
        sb.push(Kind::User, "q".into());
        sb.open_live(Kind::Agent);
        sb.append_live("par");
        sb.push(Kind::System, "after".into());
        sb.clear();
        sb.append_live("tial");
        assert_eq!(sb.entries().len(), 1);
        assert_eq!(sb.entries()[0].text, "partial");
        assert_eq!(flat(&mut sb, 40), vec!["partial"]);
    }

    #[test]
    fn empty_live_entry_is_removed_on_close() {
        let mut sb = Scrollback::default();
        sb.open_live(Kind::Agent);
        sb.push(Kind::System, "cancelled".into());
        sb.close_live();
        assert_eq!(sb.entries().len(), 1);
        assert_eq!(sb.entries()[0].kind, Kind::System);
    }

    #[test]
    fn width_change_rewraps_cached_entries() {
        let mut sb = Scrollback::default();
        sb.push(Kind::System, "aaa bbb ccc".into());
        assert_eq!(flat(&mut sb, 40).len(), 1);
        assert_eq!(flat(&mut sb, 9).len(), 2);
    }
}
