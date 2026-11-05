//! The approval gate for side-effecting tool calls.
//!
//! Per run: every call whose tool declares `needs_approval` stops and asks,
//! unless the run started with auto-approve or the user answered "a"
//! (approve this and the rest of the run) earlier in the same run. The next
//! run starts asking again.
//!
//! A decline can carry a note ("paid_by should be empty unless a name is
//! given"), which goes back to the model so it can try again differently.

use ratatui::crossterm::event::KeyCode;

/// What the UI sends back to a blocked agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verdict {
    pub decision: Decision,
    /// Why it was declined, when the user said.
    pub note: Option<String>,
}

impl Verdict {
    #[cfg(test)]
    pub fn of(decision: Decision) -> Verdict {
        Verdict {
            decision,
            note: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Approve,
    Deny,
    ApproveRest,
}

impl Decision {
    pub fn from_key(k: KeyCode) -> Option<Decision> {
        match k {
            KeyCode::Char('y') | KeyCode::Char('Y') => Some(Decision::Approve),
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => Some(Decision::Deny),
            KeyCode::Char('a') | KeyCode::Char('A') => Some(Decision::ApproveRest),
            _ => None,
        }
    }

    pub fn allows(self) -> bool {
        !matches!(self, Decision::Deny)
    }

    pub fn label(self) -> &'static str {
        match self {
            Decision::Approve => "approved",
            Decision::Deny => "declined",
            Decision::ApproveRest => "approved (and the rest of this run)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gate {
    Ask,
    /// Auto-approve: chosen up front with /auto, or with "a" mid-run.
    Open {
        auto_from_start: bool,
    },
}

impl Gate {
    pub fn new(auto: bool) -> Gate {
        if auto {
            Gate::Open {
                auto_from_start: true,
            }
        } else {
            Gate::Ask
        }
    }

    pub fn must_ask(self, needs_approval: bool) -> bool {
        needs_approval && self == Gate::Ask
    }

    pub fn record(&mut self, d: Decision) {
        if d == Decision::ApproveRest {
            *self = Gate::Open {
                auto_from_start: false,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_never_ask() {
        assert!(!Gate::new(false).must_ask(false));
    }

    #[test]
    fn writes_ask_until_approve_rest() {
        let mut g = Gate::new(false);
        assert!(g.must_ask(true));
        g.record(Decision::Approve);
        assert!(g.must_ask(true), "a single y doesn't open the gate");
        g.record(Decision::Deny);
        assert!(g.must_ask(true));
        g.record(Decision::ApproveRest);
        assert!(!g.must_ask(true));
    }

    #[test]
    fn auto_approve_skips_every_prompt() {
        assert!(!Gate::new(true).must_ask(true));
    }

    #[test]
    fn keys_map_to_decisions() {
        assert_eq!(
            Decision::from_key(KeyCode::Char('y')),
            Some(Decision::Approve)
        );
        assert_eq!(Decision::from_key(KeyCode::Esc), Some(Decision::Deny));
        assert_eq!(
            Decision::from_key(KeyCode::Char('a')),
            Some(Decision::ApproveRest)
        );
        assert_eq!(Decision::from_key(KeyCode::Char('x')), None);
        assert!(!Decision::Deny.allows());
    }
}
