//! Up/Down recall of previously submitted inputs, like a shell.

use std::fs;
use std::io::Write;
use std::path::PathBuf;

const MAX: usize = 500;

#[derive(Default)]
pub struct InputHistory {
    items: Vec<String>,
    /// Index into `items` while browsing; None when editing a fresh line.
    pos: Option<usize>,
    /// What was in the box before browsing started, restored by Down.
    draft: String,
    file: Option<PathBuf>,
}

impl InputHistory {
    /// Load from `file`, one entry per line, with `\n` inside an entry stored
    /// as the two characters `\n`.
    pub fn load(file: PathBuf) -> InputHistory {
        let items = fs::read_to_string(&file)
            .unwrap_or_default()
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| l.replace("\\n", "\n"))
            .collect::<Vec<_>>();
        let start = items.len().saturating_sub(MAX);
        InputHistory {
            items: items[start..].to_vec(),
            file: Some(file),
            ..InputHistory::default()
        }
    }

    pub fn push(&mut self, s: &str) {
        self.pos = None;
        self.draft.clear();
        let s = s.trim();
        if s.is_empty() || self.items.last().is_some_and(|l| l == s) {
            return;
        }
        self.items.push(s.to_string());
        if self.items.len() > MAX {
            self.items.remove(0);
        }
        if let Some(f) = &self.file {
            if let Some(dir) = f.parent() {
                let _ = fs::create_dir_all(dir);
            }
            if let Ok(mut fh) = fs::OpenOptions::new().create(true).append(true).open(f) {
                let _ = writeln!(fh, "{}", s.replace('\n', "\\n"));
            }
        }
    }

    /// Older entry. `current` is the box's contents, kept as the draft when
    /// browsing starts.
    pub fn up(&mut self, current: &str) -> Option<String> {
        if self.items.is_empty() {
            return None;
        }
        let next = match self.pos {
            None => {
                self.draft = current.to_string();
                self.items.len() - 1
            }
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.pos = Some(next);
        Some(self.items[next].clone())
    }

    /// Newer entry, or the draft when moving past the newest.
    pub fn down(&mut self) -> Option<String> {
        let i = self.pos?;
        if i + 1 < self.items.len() {
            self.pos = Some(i + 1);
            Some(self.items[i + 1].clone())
        } else {
            self.pos = None;
            Some(std::mem::take(&mut self.draft))
        }
    }

    pub fn browsing(&self) -> bool {
        self.pos.is_some()
    }

    pub fn reset(&mut self) {
        self.pos = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(items: &[&str]) -> InputHistory {
        let mut h = InputHistory::default();
        for i in items {
            h.push(i);
        }
        h
    }

    #[test]
    fn up_walks_back_and_down_restores_the_draft() {
        let mut h = h(&["one", "two"]);
        assert_eq!(h.up("typing").as_deref(), Some("two"));
        assert_eq!(h.up("ignored").as_deref(), Some("one"));
        assert_eq!(h.up("").as_deref(), Some("one"), "stops at the oldest");
        assert_eq!(h.down().as_deref(), Some("two"));
        assert_eq!(h.down().as_deref(), Some("typing"));
        assert_eq!(h.down(), None, "not browsing any more");
    }

    #[test]
    fn consecutive_duplicates_and_blanks_are_skipped() {
        let mut h = h(&["a", "a", "  ", "b", "a"]);
        assert_eq!(h.up("").as_deref(), Some("a"));
        assert_eq!(h.up("").as_deref(), Some("b"));
        assert_eq!(h.up("").as_deref(), Some("a"));
        assert_eq!(h.up("").as_deref(), Some("a"));
    }

    #[test]
    fn round_trips_through_the_file_including_newlines() {
        let dir = std::env::temp_dir().join(format!("amadeus-hist-{}", std::process::id()));
        let f = dir.join("input_history");
        let _ = fs::remove_file(&f);
        let mut a = InputHistory::load(f.clone());
        a.push("first");
        a.push("multi\nline");
        let mut b = InputHistory::load(f.clone());
        assert_eq!(b.up("").as_deref(), Some("multi\nline"));
        assert_eq!(b.up("").as_deref(), Some("first"));
        let _ = fs::remove_dir_all(dir);
    }
}
