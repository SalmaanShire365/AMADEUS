use std::collections::HashSet;
use std::time::Duration;

/// A slash command. `name` includes the leading slash.
pub struct Cmd {
    pub name: &'static str,
    pub args: &'static str,
    pub help: &'static str,
}

pub const COMMANDS: &[Cmd] = &[
    Cmd { name: "/rag",     args: "<question>", help: "codebase-grounded answer" },
    Cmd { name: "/ask",     args: "<task>",     help: "ask the model with no retrieval" },
    Cmd { name: "/index",   args: "[dir]",      help: "build or refresh the RAG index" },
    Cmd { name: "/file",    args: "<path>",     help: "stage a file into the next prompt" },
    Cmd { name: "/shell",   args: "<cmd>",      help: "run a command, output into scrollback" },
    Cmd { name: "/agent", args: "<task>", help: "autonomous coding agent" },
    Cmd { name: "/model",   args: "",           help: "cycle fast / main / heavy" },
    Cmd { name: "/history", args: "",           help: "show this session's history" },
    Cmd { name: "/clear",   args: "",           help: "clear scrollback and history file" },
    Cmd { name: "/quit",    args: "",           help: "exit" },
];

/// Keybinding reminders, mixed into the idle rotation alongside unused commands.
const TIPS: &[&str] = &[
    "ctrl+t cycles the model tier",
    "alt+enter inserts a newline, enter sends",
    "pgup / pgdn scroll, end re-pins to the bottom",
    "ctrl+c cancels a running answer",
    "type / to see every command",
];

/// How long after the last keystroke the bar stays out of the way.
const QUIET: Duration = Duration::from_millis(800);
/// How long each idle hint is shown before rotating.
const ROTATE: Duration = Duration::from_secs(4);

#[derive(Debug, PartialEq)]
pub enum Hint {
    Hidden,
    Tip(String),
    /// Completion candidates, rendered as a popup above the input.
    Menu(Vec<&'static Cmd>),
}

impl std::fmt::Debug for Cmd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cmd({})", self.name)
    }
}

impl PartialEq for Cmd {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

fn usage(c: &Cmd) -> String {
    if c.args.is_empty() {
        format!("{}  — {}", c.name, c.help)
    } else {
        format!("{} {}  — {}", c.name, c.args, c.help)
    }
}

/// Candidates for a partially typed command token.
pub fn matches(token: &str) -> Vec<&'static Cmd> {
    COMMANDS.iter().filter(|c| c.name.starts_with(token)).collect()
}

/// The whole hint bar, as a pure function of input state. No terminal, no
/// clock reads, so it is testable directly.
///
/// `since_key` is time since the last keystroke; `idle_for` is time since the
/// input buffer last became empty (used only to rotate idle hints).
pub fn hint(
    input: &str,
    since_key: Duration,
    idle_for: Duration,
    used: &HashSet<String>,
) -> Hint {
    if input.starts_with('/') {
        let token = input.split_whitespace().next().unwrap_or("/");
        // Still typing the command itself: offer completions.
        if input.len() == token.len() {
            let m = matches(token);
            return if m.is_empty() {
                Hint::Tip(format!("no command matches {token}"))
            } else {
                Hint::Menu(m)
            };
        }
        // Command is settled, user is typing arguments: show its usage.
        return match COMMANDS.iter().find(|c| c.name == token) {
            Some(c) => Hint::Tip(usage(c)),
            None => Hint::Tip(format!("unknown command {token}")),
        };
    }

    if since_key < QUIET {
        return Hint::Hidden;
    }

    if input.trim().is_empty() {
        let mut pool: Vec<String> = COMMANDS
            .iter()
            .filter(|c| !used.contains(c.name))
            .map(usage)
            .collect();
        pool.extend(TIPS.iter().map(|t| t.to_string()));
        if pool.is_empty() {
            return Hint::Hidden;
        }
        let slot = (idle_for.as_millis() / ROTATE.as_millis()) as usize % pool.len();
        return Hint::Tip(pool[slot].clone());
    }

    Hint::Tip("enter to send · alt+enter for a newline".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn none() -> HashSet<String> {
        HashSet::new()
    }

    #[test]
    fn slash_prefix_offers_completions() {
        let h = hint("/r", Duration::ZERO, Duration::ZERO, &none());
        match h {
            Hint::Menu(m) => assert!(m.iter().any(|c| c.name == "/rag")),
            other => panic!("expected a menu, got {other:?}"),
        }
    }

    #[test]
    fn completions_appear_immediately_even_mid_keystroke() {
        // The quiet window must not suppress the command menu.
        assert!(matches!(
            hint("/ind", Duration::from_millis(5), Duration::ZERO, &none()),
            Hint::Menu(_)
        ));
    }

    #[test]
    fn typing_arguments_shows_usage() {
        let h = hint("/rag where is chunking", Duration::ZERO, Duration::ZERO, &none());
        match h {
            Hint::Tip(t) => assert!(t.contains("grounded")),
            other => panic!("expected usage, got {other:?}"),
        }
    }

    #[test]
    fn unknown_command_is_named() {
        let h = hint("/nope ", Duration::ZERO, Duration::ZERO, &none());
        assert_eq!(h, Hint::Tip("unknown command /nope".into()));
    }

    #[test]
    fn hidden_while_actively_typing_prose() {
        assert_eq!(
            hint("how does chunk", Duration::from_millis(120), Duration::ZERO, &none()),
            Hint::Hidden
        );
    }

    #[test]
    fn prose_hint_returns_after_the_quiet_window() {
        assert!(matches!(
            hint("how does chunk", Duration::from_secs(2), Duration::ZERO, &none()),
            Hint::Tip(_)
        ));
    }

    #[test]
    fn idle_hints_rotate_on_the_four_second_boundary() {
        let a = hint("", Duration::from_secs(3), Duration::from_secs(1), &none());
        let b = hint("", Duration::from_secs(3), Duration::from_secs(3), &none());
        let c = hint("", Duration::from_secs(3), Duration::from_secs(5), &none());
        assert_eq!(a, b);
        assert_ne!(b, c);
    }

    #[test]
    fn used_commands_drop_out_of_the_rotation() {
        let mut used = HashSet::new();
        for c in COMMANDS {
            used.insert(c.name.to_string());
        }
        // Only keybinding tips remain, so no hint should name a command.
        for secs in 0..40 {
            if let Hint::Tip(t) = hint("", Duration::from_secs(3), Duration::from_secs(secs), &used)
            {
                assert!(!t.starts_with('/'), "still advertising a used command: {t}");
            }
        }
    }
}