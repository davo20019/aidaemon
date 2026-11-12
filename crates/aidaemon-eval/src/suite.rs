//! Running many scenarios many times.

use std::path::Path;

use futures::stream::{self, StreamExt};
use serde::de::DeserializeOwned;

use crate::report::Report;
use crate::runner::{AgentFactory, AgentSubject, CaseResult, Subject};
use crate::scenario::{load_dir, EvalError, NoExtension, Scenario};

/// A set of scenarios run N times each.
#[derive(Debug, Clone)]
pub struct Suite<S = NoExtension, X = NoExtension> {
    pub scenarios: Vec<Scenario<S, X>>,
    parallelism: usize,
}

impl Suite {
    /// Load every `*.yaml` / `*.yml` file in `dir`, sorted by file name.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, EvalError> {
        Self::load_dir(dir)
    }

    /// Run every scenario `n_runs` times and return raw results, grouped per
    /// scenario in suite order (runs within a group in run order).
    pub async fn run_cases(
        &self,
        factory: &dyn AgentFactory,
        n_runs: usize,
    ) -> Vec<Vec<CaseResult>> {
        self.run_subject_cases(&AgentSubject::new(factory), n_runs)
            .await
    }

    /// Run every scenario `n_runs` times and aggregate into a [`Report`].
    pub async fn run(&self, factory: &dyn AgentFactory, n_runs: usize) -> Report {
        self.run_subject(&AgentSubject::new(factory), n_runs).await
    }
}

impl<S, X> Suite<S, X>
where
    S: DeserializeOwned,
    X: DeserializeOwned + Default,
{
    /// [`Suite::from_dir`] with the subject's `setup:` and `expect.ext:`
    /// types.
    pub fn load_dir(dir: impl AsRef<Path>) -> Result<Self, EvalError> {
        Ok(Self::new(load_dir(dir.as_ref())?))
    }
}

impl<S, X> Suite<S, X> {
    pub fn new(scenarios: Vec<Scenario<S, X>>) -> Self {
        Self {
            scenarios,
            parallelism: 4,
        }
    }

    /// Maximum runs in flight at once (default 4, minimum 1). Mind provider
    /// rate limits when running against a live model.
    pub fn parallelism(mut self, limit: usize) -> Self {
        self.parallelism = limit.max(1);
        self
    }

    /// Run every scenario against the factory's live provider, scripted or
    /// not: see [`Scenario::into_live`].
    pub fn live(mut self) -> Self {
        self.scenarios = self
            .scenarios
            .into_iter()
            .map(Scenario::into_live)
            .collect();
        self
    }

    /// Keep only scenarios whose name contains `pattern`.
    pub fn filter(mut self, pattern: &str) -> Self {
        self.scenarios.retain(|s| s.name.contains(pattern));
        self
    }

    /// [`Suite::run_cases`] for any [`Subject`].
    pub async fn run_subject_cases<T>(&self, subject: &T, n_runs: usize) -> Vec<Vec<CaseResult>>
    where
        T: Subject<Setup = S, ExpectExt = X>,
    {
        let n_runs = n_runs.max(1);
        let jobs = self
            .scenarios
            .iter()
            .enumerate()
            .flat_map(|(i, s)| (0..n_runs).map(move |run| (i, run, s)));
        let mut done: Vec<(usize, usize, CaseResult)> = stream::iter(jobs)
            .map(|(i, run, scenario)| async move { (i, run, scenario.run_on(subject, run).await) })
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

    /// [`Suite::run`] for any [`Subject`].
    pub async fn run_subject<T>(&self, subject: &T, n_runs: usize) -> Report
    where
        T: Subject<Setup = S, ExpectExt = X>,
    {
        let grouped = self.run_subject_cases(subject, n_runs).await;
        Report::from_results(&grouped)
    }
}
