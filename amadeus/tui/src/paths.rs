//! Where things live.
//!
//! AMADEUS_HOME (the checkout: amadeus.sh, rag/, templates/) has one rule,
//! shared with amadeus.sh: it's the directory the running code came from.
//! The launcher takes it from its own resolved path and exports it; the TUI
//! takes it from its executable's location (…/tui/target/release/amadeus-tui)
//! and only falls back to $AMADEUS_HOME when the binary was copied out of a
//! checkout. Both resolve to the same directory by construction.
//!
//! Per-user state goes under $XDG_STATE_HOME/amadeus (default
//! ~/.local/state/amadeus), never into the project being worked on.

use std::path::{Path, PathBuf};

fn is_checkout(p: &Path) -> bool {
    p.join("amadeus.sh").is_file() && p.join("rag/rag.py").is_file()
}

pub fn amadeus_home() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        let exe = exe.canonicalize().unwrap_or(exe);
        if let Some(p) = exe.ancestors().skip(1).take(5).find(|p| is_checkout(p)) {
            return p.to_path_buf();
        }
    }
    if let Ok(home) = std::env::var("AMADEUS_HOME") {
        return PathBuf::from(home);
    }
    PathBuf::from(".")
}

pub fn rag_script() -> PathBuf {
    amadeus_home().join("rag/rag.py")
}

/// AMADEUS's own Python, never the project's.
pub fn python() -> PathBuf {
    let venv = amadeus_home().join("venv/bin/python3");
    if venv.is_file() {
        venv
    } else {
        PathBuf::from("python3")
    }
}

pub fn state_dir() -> PathBuf {
    state_dir_from(
        std::env::var_os("XDG_STATE_HOME").map(PathBuf::from),
        std::env::var_os("HOME").map(PathBuf::from),
    )
}

fn state_dir_from(xdg: Option<PathBuf>, home: Option<PathBuf>) -> PathBuf {
    // The XDG spec says relative values are invalid and must be ignored.
    match xdg.filter(|p| p.is_absolute()) {
        Some(x) => x.join("amadeus"),
        None => home
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join(".local/state/amadeus"),
    }
}

/// 64-bit FNV-1a. Stable across Rust versions, unlike `DefaultHasher`.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// A readable, collision-safe directory name for a project path:
/// `home_me_notes-1a2b3c4d5e6f7a8b`.
pub fn project_slug(root: &Path) -> String {
    let s = root.to_string_lossy();
    let flat: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let flat = flat.trim_matches('_');
    let tail: String = {
        let chars: Vec<char> = flat.chars().collect();
        chars[chars.len().saturating_sub(48)..].iter().collect()
    };
    format!(
        "{}-{:016x}",
        tail.trim_start_matches('_'),
        fnv1a(s.as_bytes())
    )
}

pub fn project_dir(root: &Path) -> PathBuf {
    state_dir().join("projects").join(project_slug(root))
}

pub fn history_file(root: &Path) -> PathBuf {
    project_dir(root).join("history.txt")
}

pub fn runs_dir(root: &Path) -> PathBuf {
    project_dir(root).join("runs")
}

pub fn input_history_file() -> PathBuf {
    state_dir().join("input_history")
}

/// User templates, alongside the shipped ones in AMADEUS_HOME/templates.
pub fn user_templates_dir() -> PathBuf {
    let base = match std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from) {
        Some(x) if x.is_absolute() => x,
        _ => std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join(".config"),
    };
    base.join("amadeus/templates")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xdg_state_home_wins_when_absolute() {
        assert_eq!(
            state_dir_from(Some("/x/state".into()), Some("/home/u".into())),
            PathBuf::from("/x/state/amadeus")
        );
    }

    #[test]
    fn relative_xdg_is_ignored() {
        assert_eq!(
            state_dir_from(Some("rel".into()), Some("/home/u".into())),
            PathBuf::from("/home/u/.local/state/amadeus")
        );
        assert_eq!(
            state_dir_from(None, Some("/home/u".into())),
            PathBuf::from("/home/u/.local/state/amadeus")
        );
    }

    #[test]
    fn slug_is_readable_stable_and_distinguishes_lookalikes() {
        let a = project_slug(Path::new("/home/u/a/b"));
        let b = project_slug(Path::new("/home/u/a_b"));
        assert!(a.starts_with("home_u_a_b-"), "{a}");
        assert!(b.starts_with("home_u_a_b-"), "{b}");
        assert_ne!(a, b, "hash must separate paths that flatten the same");
        assert_eq!(a, project_slug(Path::new("/home/u/a/b")));
    }

    #[test]
    fn history_never_lands_in_the_project() {
        let root = Path::new("/home/u/project");
        let h = history_file(root);
        assert!(!h.starts_with(root), "{}", h.display());
        assert!(h.ends_with("history.txt"));
    }

    #[test]
    fn fnv_known_vector() {
        // Published FNV-1a 64 test vector.
        assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
    }
}
