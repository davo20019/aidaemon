use std::fmt;

/// The outcome of a conformance kit: which checks passed, which failed and
/// why, plus informational notes (for example a check that was vacuous for
/// this implementation).
///
/// Check names are stable, so tests may assert on individual checks with
/// [`ConformanceReport::passed_check`] / [`ConformanceReport::failure`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConformanceReport {
    /// What was checked (a tool name, provider name, channel name, ...).
    pub subject: String,
    /// Names of the checks that passed, in run order.
    pub passed: Vec<String>,
    /// `(check name, reason)` for every failed check, in run order.
    pub failed: Vec<(String, String)>,
    /// Informational remarks that do not affect the verdict.
    pub notes: Vec<String>,
}

impl ConformanceReport {
    pub fn new(subject: impl Into<String>) -> Self {
        Self {
            subject: subject.into(),
            ..Self::default()
        }
    }

    /// Record a passed check.
    pub fn pass(&mut self, check: impl Into<String>) {
        self.passed.push(check.into());
    }

    /// Record a failed check.
    pub fn fail(&mut self, check: impl Into<String>, reason: impl Into<String>) {
        self.failed.push((check.into(), reason.into()));
    }

    /// Record `Ok` as a pass and `Err(reason)` as a failure.
    pub fn record(&mut self, check: impl Into<String>, result: Result<(), String>) {
        match result {
            Ok(()) => self.pass(check),
            Err(reason) => self.fail(check, reason),
        }
    }

    /// Add an informational note.
    pub fn note(&mut self, note: impl Into<String>) {
        self.notes.push(note.into());
    }

    /// True when no check failed.
    pub fn is_ok(&self) -> bool {
        self.failed.is_empty()
    }

    /// True when `check` ran and passed.
    pub fn passed_check(&self, check: &str) -> bool {
        self.passed.iter().any(|c| c == check)
    }

    /// The failure reason for `check`, if it failed.
    pub fn failure(&self, check: &str) -> Option<&str> {
        self.failed
            .iter()
            .find(|(c, _)| c == check)
            .map(|(_, r)| r.as_str())
    }

    /// Append another report's results (for example one per tool) to this
    /// one. Check names from `other` are prefixed with its subject when it
    /// differs from this report's.
    pub fn merge(&mut self, other: ConformanceReport) {
        let prefix = if other.subject.is_empty() || other.subject == self.subject {
            String::new()
        } else {
            format!("{}: ", other.subject)
        };
        self.passed
            .extend(other.passed.into_iter().map(|c| format!("{prefix}{c}")));
        self.failed.extend(
            other
                .failed
                .into_iter()
                .map(|(c, r)| (format!("{prefix}{c}"), r)),
        );
        self.notes
            .extend(other.notes.into_iter().map(|n| format!("{prefix}{n}")));
    }

    /// Panic if any check failed, listing every failure (and the notes) in
    /// the message.
    #[track_caller]
    pub fn assert_all_passed(&self) {
        if !self.is_ok() {
            panic!("{self}");
        }
    }
}

impl fmt::Display for ConformanceReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let subject = if self.subject.is_empty() {
            "conformance"
        } else {
            self.subject.as_str()
        };
        writeln!(
            f,
            "{subject}: {} passed, {} failed",
            self.passed.len(),
            self.failed.len()
        )?;
        for (check, reason) in &self.failed {
            writeln!(f, "  FAIL {check}: {reason}")?;
        }
        for note in &self.notes {
            writeln!(f, "  note: {note}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_prefixes_and_assert_lists_every_failure() {
        let mut all = ConformanceReport::new("tools");
        let mut one = ConformanceReport::new("read_file");
        one.pass("a");
        one.fail("b", "broken");
        let mut two = ConformanceReport::new("shell");
        two.fail("c", "also broken");
        all.merge(one);
        all.merge(two);
        assert_eq!(all.passed, ["read_file: a"]);
        assert_eq!(all.failure("shell: c"), Some("also broken"));
        let message = std::panic::catch_unwind(|| all.assert_all_passed())
            .unwrap_err()
            .downcast::<String>()
            .unwrap();
        assert!(message.contains("read_file: b: broken"), "{message}");
        assert!(message.contains("shell: c: also broken"), "{message}");
    }
}
