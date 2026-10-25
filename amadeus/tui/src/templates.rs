//! Agent templates: plain TOML files that set the system prompt, the tools a
//! run may use, the default model tier and an example first message.
//!
//! Shipped templates live in $AMADEUS_HOME/templates; files in
//! ~/.config/amadeus/templates with the same `name` replace them.

use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::app::Tier;
use crate::paths;
use crate::tools;

#[derive(Debug, Clone, Deserialize)]
pub struct Template {
    pub name: String,
    pub title: String,
    pub description: String,
    pub system: String,
    pub tools: Vec<String>,
    #[serde(default = "default_tier")]
    pub tier: String,
    #[serde(default)]
    pub example: String,
    #[serde(default = "default_steps")]
    pub max_steps: usize,
    /// Position in the picker; lower first, then by name.
    #[serde(default = "default_order")]
    pub order: u32,
}

fn default_order() -> u32 {
    100
}

fn default_tier() -> String {
    "main".into()
}

fn default_steps() -> usize {
    16
}

impl Template {
    pub fn tier(&self) -> Tier {
        match self.tier.as_str() {
            "fast" => Tier::Fast,
            "heavy" => Tier::Heavy,
            _ => Tier::Main,
        }
    }

    pub fn uses_approval(&self) -> bool {
        self.tools
            .iter()
            .any(|t| tools::spec(t).is_some_and(|s| s.needs_approval))
    }
}

pub fn parse(text: &str) -> Result<Template, String> {
    let t: Template = toml::from_str(text).map_err(|e| e.to_string())?;
    if t.name.trim().is_empty() || t.name.contains(char::is_whitespace) {
        return Err(format!("name {:?} must be one word", t.name));
    }
    if let Some(bad) = t.tools.iter().find(|n| tools::spec(n).is_none()) {
        let known: Vec<&str> = tools::TOOLS.iter().map(|t| t.name).collect();
        return Err(format!(
            "unknown tool {bad:?} (known: {})",
            known.join(", ")
        ));
    }
    if !matches!(t.tier.as_str(), "fast" | "main" | "heavy") {
        return Err(format!("tier {:?} must be fast, main or heavy", t.tier));
    }
    Ok(t)
}

fn load_dir(dir: &Path, out: &mut Vec<Template>, errors: &mut Vec<String>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    let mut files: Vec<_> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "toml"))
        .collect();
    files.sort();
    for f in files {
        match fs::read_to_string(&f)
            .map_err(|e| e.to_string())
            .and_then(|s| parse(&s))
        {
            Ok(t) => {
                out.retain(|o| o.name != t.name);
                out.push(t);
            }
            Err(e) => errors.push(format!("{}: {e}", f.display())),
        }
    }
}

/// Every template, shipped first then user overrides, plus any load errors.
pub fn load_all() -> (Vec<Template>, Vec<String>) {
    let (mut out, mut errors) = (Vec::new(), Vec::new());
    load_dir(
        &paths::amadeus_home().join("templates"),
        &mut out,
        &mut errors,
    );
    load_dir(&paths::user_templates_dir(), &mut out, &mut errors);
    out.sort_by(|a, b| a.order.cmp(&b.order).then_with(|| a.name.cmp(&b.name)));
    (out, errors)
}

#[cfg(test)]
mod tests {
    use super::*;

    const OK: &str = r#"
name = "t"
title = "T"
description = "d"
system = "s"
tools = ["read_file", "write_file"]
tier = "fast"
"#;

    #[test]
    fn parses_with_defaults() {
        let t = parse(OK).unwrap();
        assert_eq!(t.tier(), Tier::Fast);
        assert_eq!(t.max_steps, 16);
        assert!(t.uses_approval());
    }

    #[test]
    fn unknown_tools_are_rejected_with_the_known_list() {
        let e = parse(&OK.replace("write_file", "delete_everything")).unwrap_err();
        assert!(
            e.contains("delete_everything") && e.contains("read_file"),
            "{e}"
        );
    }

    #[test]
    fn bad_tier_is_rejected() {
        assert!(parse(&OK.replace("\"fast\"", "\"huge\"")).is_err());
    }

    #[test]
    fn shipped_templates_all_parse() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../templates");
        let (mut out, mut errors) = (Vec::new(), Vec::new());
        load_dir(&dir, &mut out, &mut errors);
        assert!(errors.is_empty(), "{errors:?}");
        let names: Vec<&str> = out.iter().map(|t| t.name.as_str()).collect();
        for want in [
            "folder-summary",
            "notes-to-spreadsheet",
            "draft-replies",
            "code-helper",
        ] {
            assert!(names.contains(&want), "missing {want}: {names:?}");
        }
        for t in &out {
            assert!(!t.example.is_empty(), "{} needs an example", t.name);
        }
    }
}
