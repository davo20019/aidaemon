//! N-of-M aggregation, reports and baseline comparison.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::runner::CaseResult;
use crate::scenario::EvalError;

/// One failure message and how many runs produced it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FailureCount {
    pub message: String,
    pub count: usize,
}

/// Aggregated results for one scenario over N runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenarioReport {
    pub name: String,
    pub runs: usize,
    pub passes: usize,
    /// `passes / runs`, in `[0, 1]`.
    pub pass_rate: f64,
    pub mean_turns: f64,
    pub mean_billable_tokens: f64,
    pub mean_duration_ms: f64,
    /// Distinct failure messages, most frequent first.
    pub failures: Vec<FailureCount>,
}

/// The result of [`crate::Suite::run`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub total_runs: usize,
    pub total_passes: usize,
    pub pass_rate: f64,
    pub scenarios: Vec<ScenarioReport>,
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, n) = values.fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
    if n == 0 { 0.0 } else { sum / n as f64 }
}

fn ratio(passes: usize, runs: usize) -> f64 {
    if runs == 0 {
        0.0
    } else {
        passes as f64 / runs as f64
    }
}

impl ScenarioReport {
    /// Aggregate the runs of one scenario. `name` is used when `runs` is
    /// empty; otherwise the runs' scenario name wins.
    pub fn from_runs(name: &str, runs: &[CaseResult]) -> Self {
        let passes = runs.iter().filter(|r| r.passed).count();
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for failure in runs.iter().flat_map(|r| r.failures.iter()) {
            *counts.entry(failure.as_str()).or_default() += 1;
        }
        let mut failures: Vec<FailureCount> = counts
            .into_iter()
            .map(|(message, count)| FailureCount {
                message: message.to_string(),
                count,
            })
            .collect();
        // Stable: BTreeMap order breaks ties alphabetically.
        failures.sort_by_key(|f| std::cmp::Reverse(f.count));
        Self {
            name: runs
                .first()
                .map(|r| r.scenario.clone())
                .unwrap_or_else(|| name.to_string()),
            runs: runs.len(),
            passes,
            pass_rate: ratio(passes, runs.len()),
            mean_turns: mean(runs.iter().map(|r| f64::from(r.turns))),
            mean_billable_tokens: mean(runs.iter().map(|r| r.billable_tokens as f64)),
            mean_duration_ms: mean(runs.iter().map(|r| r.duration_ms)),
            failures,
        }
    }
}

impl Report {
    /// Aggregate raw results grouped per scenario (as from
    /// [`crate::Suite::run_cases`]). Empty groups are skipped.
    pub fn from_results(grouped: &[Vec<CaseResult>]) -> Self {
        let scenarios: Vec<ScenarioReport> = grouped
            .iter()
            .filter(|runs| !runs.is_empty())
            .map(|runs| ScenarioReport::from_runs("", runs))
            .collect();
        let total_runs = scenarios.iter().map(|s| s.runs).sum();
        let total_passes = scenarios.iter().map(|s| s.passes).sum();
        Self {
            total_runs,
            total_passes,
            pass_rate: ratio(total_passes, total_runs),
            scenarios,
        }
    }

    pub fn scenario(&self, name: &str) -> Option<&ScenarioReport> {
        self.scenarios.iter().find(|s| s.name == name)
    }

    /// True when every run of every scenario passed.
    pub fn all_passed(&self) -> bool {
        self.total_runs > 0 && self.total_passes == self.total_runs
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("report serializes")
    }

    /// Load a report saved with [`Report::to_json`], e.g. a baseline.
    pub fn from_json(json: &str) -> Result<Self, EvalError> {
        serde_json::from_str(json).map_err(|e| EvalError::ReportJson(e.to_string()))
    }

    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "# Eval report\n");
        let _ = writeln!(
            out,
            "{} scenario(s), {} run(s). Overall pass rate: {} ({}/{})\n",
            self.scenarios.len(),
            self.total_runs,
            pct(self.pass_rate),
            self.total_passes,
            self.total_runs
        );
        let _ = writeln!(
            out,
            "| Scenario | Pass rate | Passed | Mean turns | Mean billable tokens | Mean duration (ms) |"
        );
        let _ = writeln!(out, "|---|---:|---:|---:|---:|---:|");
        for s in &self.scenarios {
            let _ = writeln!(
                out,
                "| {} | {} | {}/{} | {:.1} | {:.0} | {:.1} |",
                cell(&s.name),
                pct(s.pass_rate),
                s.passes,
                s.runs,
                s.mean_turns,
                s.mean_billable_tokens,
                s.mean_duration_ms
            );
        }
        let failing: Vec<&ScenarioReport> = self
            .scenarios
            .iter()
            .filter(|s| !s.failures.is_empty())
            .collect();
        if !failing.is_empty() {
            let _ = writeln!(out, "\n## Failures");
            for s in failing {
                let _ = writeln!(out, "\n### {}\n", s.name);
                for f in &s.failures {
                    let _ = writeln!(out, "- {}x {}", f.count, f.message);
                }
            }
        }
        out
    }
}

/// One row of a [`Comparison`]. Rates are `None` when the scenario is
/// missing from that report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComparisonRow {
    pub scenario: String,
    pub baseline_pass_rate: Option<f64>,
    pub candidate_pass_rate: Option<f64>,
    /// `candidate - baseline`, when both exist.
    pub delta: Option<f64>,
    pub baseline_runs: usize,
    pub candidate_runs: usize,
}

/// Pass-rate deltas between two reports (policy on vs off, old vs new loop).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Comparison {
    pub baseline_pass_rate: f64,
    pub candidate_pass_rate: f64,
    pub delta: f64,
    /// Baseline order first, then scenarios only in the candidate.
    pub rows: Vec<ComparisonRow>,
}

/// Compare two reports scenario by scenario.
pub fn compare(baseline: &Report, candidate: &Report) -> Comparison {
    let mut rows = Vec::new();
    for b in &baseline.scenarios {
        let c = candidate.scenario(&b.name);
        rows.push(ComparisonRow {
            scenario: b.name.clone(),
            baseline_pass_rate: Some(b.pass_rate),
            candidate_pass_rate: c.map(|c| c.pass_rate),
            delta: c.map(|c| c.pass_rate - b.pass_rate),
            baseline_runs: b.runs,
            candidate_runs: c.map_or(0, |c| c.runs),
        });
    }
    for c in &candidate.scenarios {
        if baseline.scenario(&c.name).is_none() {
            rows.push(ComparisonRow {
                scenario: c.name.clone(),
                baseline_pass_rate: None,
                candidate_pass_rate: Some(c.pass_rate),
                delta: None,
                baseline_runs: 0,
                candidate_runs: c.runs,
            });
        }
    }
    Comparison {
        baseline_pass_rate: baseline.pass_rate,
        candidate_pass_rate: candidate.pass_rate,
        delta: candidate.pass_rate - baseline.pass_rate,
        rows,
    }
}

impl Comparison {
    /// Rows whose pass rate dropped.
    pub fn regressions(&self) -> impl Iterator<Item = &ComparisonRow> {
        self.rows
            .iter()
            .filter(|r| r.delta.is_some_and(|d| d < 0.0))
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("comparison serializes")
    }

    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "# Eval comparison\n");
        let _ = writeln!(
            out,
            "Overall pass rate: {} -> {} ({})\n",
            pct(self.baseline_pass_rate),
            pct(self.candidate_pass_rate),
            signed_pct(self.delta)
        );
        let _ = writeln!(out, "| Scenario | Baseline | Candidate | Delta |");
        let _ = writeln!(out, "|---|---:|---:|---:|");
        for r in &self.rows {
            let side = |rate: Option<f64>, runs: usize| match rate {
                Some(rate) => format!("{} (n={runs})", pct(rate)),
                None => "-".to_string(),
            };
            let _ = writeln!(
                out,
                "| {} | {} | {} | {} |",
                cell(&r.scenario),
                side(r.baseline_pass_rate, r.baseline_runs),
                side(r.candidate_pass_rate, r.candidate_runs),
                r.delta.map_or_else(|| "-".to_string(), signed_pct)
            );
        }
        out
    }
}

fn pct(rate: f64) -> String {
    format!("{:.1}%", rate * 100.0)
}

fn signed_pct(delta: f64) -> String {
    let points = delta * 100.0;
    if points.abs() < 0.05 {
        "0.0 pp".to_string()
    } else {
        format!("{points:+.1} pp")
    }
}

fn cell(text: &str) -> String {
    text.replace('|', "\\|")
}
