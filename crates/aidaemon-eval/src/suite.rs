//! Running many scenarios many times.

use std::path::Path;

use futures::stream::{self, StreamExt};

use crate::report::Report;
use crate::runner::{AgentFactory, CaseResult};
use crate::scenario::{load_dir, EvalError, Scenario};

/// A set of scenarios run N times each.
#[derive(Debug, Clone)]
pub struct Suite {
    pub scenarios: Vec<Scenario>,
    parallelism: usize,
}

impl Suite {
    pub fn new(scenarios: Vec<Scenario>) -> Self {
        Self {
            scenarios,
            parallelism: 4,
        }
    }

    /// Load every `*.yaml` / `*.yml` file in `dir`, sorted by file name.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, EvalError> {
        Ok(Self::new(load_dir(dir.as_ref())?))
    }

    /// Maximum runs in flight at once (default 4, minimum 1). Mind provider
    /// rate limits when running against a live model.
    pub fn parallelism(mut self, limit: usize) -> Self {
        self.parallelism = limit.max(1);
        self
    }

    /// Keep only scenarios whose name contains `pattern`.
    pub fn filter(mut self, pattern: &str) -> Self {
        self.scenarios.retain(|s| s.name.contains(pattern));
        self
    }

    /// Run every scenario `n_runs` times and return raw results, grouped per
    /// scenario in suite order (runs within a group in run order).
    pub async fn run_cases(
        &self,
        factory: &dyn AgentFactory,
        n_runs: usize,
    ) -> Vec<Vec<CaseResult>> {
        let n_runs = n_runs.max(1);
        let jobs = self
            .scenarios
            .iter()
            .enumerate()
            .flat_map(|(i, s)| (0..n_runs).map(move |run| (i, run, s)));
        let mut done: Vec<(usize, usize, CaseResult)> =
            stream::iter(jobs)
                .map(|(i, run, scenario)| async move {
                    (i, run, scenario.run_numbered(factory, run).await)
                })
                .buffer_unordered(self.parallelism)
                .collect()
                .await;
        done.sort_by_key(|(i, run, _)| (*i, *run));
        let mut grouped: Vec<Vec<CaseResult>> = vec![Vec::new(); self.scenarios.len()];
        for (i, _, result) in done {
            grouped[i].push(result);
        }
        grouped
    }

    /// Run every scenario `n_runs` times and aggregate into a [`Report`].
    pub async fn run(&self, factory: &dyn AgentFactory, n_runs: usize) -> Report {
        let grouped = self.run_cases(factory, n_runs).await;
        Report::from_results(&grouped)
    }
}
