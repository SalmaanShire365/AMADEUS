//! Is Ollama up, and are the configured models pulled? Run once at startup
//! (on a background thread) and by `amadeus doctor`.

use crate::ollama;

#[derive(Debug, Clone)]
pub struct ModelStatus {
    pub tier: &'static str,
    pub name: String,
    pub pulled: bool,
}

#[derive(Debug, Clone)]
pub struct Health {
    pub url: String,
    pub version: Result<String, String>,
    pub models: Vec<ModelStatus>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Ollama reachable, every configured model pulled.
    Ok,
    /// Reachable, but at least one tier's model is missing.
    Degraded,
    Down,
}

impl Health {
    pub fn status(&self) -> Status {
        if self.version.is_err() {
            Status::Down
        } else if self.models.iter().all(|m| m.pulled) {
            Status::Ok
        } else {
            Status::Degraded
        }
    }

    pub fn missing(&self) -> impl Iterator<Item = &ModelStatus> {
        self.models.iter().filter(|m| !m.pulled)
    }

    /// One line per problem, each with the command that fixes it. Empty when
    /// healthy.
    pub fn problems(&self) -> Vec<String> {
        if let Err(e) = &self.version {
            return vec![format!(
                "Ollama isn't answering at {} ({e}). Start it with: ollama serve",
                self.url
            )];
        }
        self.missing()
            .map(|m| {
                format!(
                    "{} model {} isn't pulled. Fix: ollama pull {}",
                    m.tier, m.name, m.name
                )
            })
            .collect()
    }
}

pub fn check(url: &str, tiers: &[(&'static str, String)]) -> Health {
    let version = ollama::version(url);
    let pulled = if version.is_ok() {
        ollama::tags(url).unwrap_or_default()
    } else {
        Vec::new()
    };
    let mut models: Vec<ModelStatus> = Vec::new();
    for (tier, name) in tiers {
        models.push(ModelStatus {
            tier,
            name: name.clone(),
            pulled: ollama::is_pulled(&pulled, name),
        });
    }
    Health {
        url: url.to_string(),
        version,
        models,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(version: Result<&str, &str>, pulled: &[bool]) -> Health {
        Health {
            url: "http://x".into(),
            version: version.map(String::from).map_err(String::from),
            models: pulled
                .iter()
                .enumerate()
                .map(|(i, p)| ModelStatus {
                    tier: ["fast", "main", "heavy"][i],
                    name: format!("m{i}"),
                    pulled: *p,
                })
                .collect(),
        }
    }

    #[test]
    fn down_reports_only_the_server() {
        let x = h(Err("refused"), &[false, false]);
        assert_eq!(x.status(), Status::Down);
        let p = x.problems();
        assert_eq!(p.len(), 1);
        assert!(p[0].contains("ollama serve"));
    }

    #[test]
    fn missing_models_come_with_a_pull_command() {
        let x = h(Ok("0.40"), &[true, false, true]);
        assert_eq!(x.status(), Status::Degraded);
        assert_eq!(
            x.problems(),
            vec!["main model m1 isn't pulled. Fix: ollama pull m1"]
        );
    }

    #[test]
    fn all_pulled_is_ok() {
        let x = h(Ok("0.40"), &[true, true, true]);
        assert_eq!(x.status(), Status::Ok);
        assert!(x.problems().is_empty());
    }
}
