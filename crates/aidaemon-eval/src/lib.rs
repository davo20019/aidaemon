//! # aidaemon-eval (experimental)
//!
//! An eval kit for agents built on [`aidaemon_core`]. Model output is
//! stochastic, so a single pass or failure means little: run each scenario N
//! times, read pass *rates*, and assert on trajectories (which tools ran, in
//! what order, with what receipt status) rather than only on final text.
//!
//! - [`Scenario`]: one YAML case. Scripted scenarios (`script:`) replay fixed
//!   model replies and are deterministic harness tests; unscripted ones run
//!   against a live provider.
//! - [`AgentFactory`]: builds the agent under test for each run.
//! - [`Subject`]: the seam behind it. Anything that can run a scenario and
//!   report a neutral [`Trajectory`] can be evaluated.
//! - [`Suite`]: loads a directory of scenarios and runs each N times with
//!   bounded concurrency, producing a [`Report`] (Markdown or JSON).
//! - [`compare`]: pass-rate deltas between two reports, for policy on/off or
//!   old-vs-new loop experiments.
//!
//! ## Scenario format
//!
//! ```yaml
//! name: weather_round_trip
//! input: What's the weather in Springfield?
//! instructions: Answer briefly.          # optional
//! tags: [weather]                        # optional, for the factory
//! limits: { max_turns: 5 }               # optional: max_turns, max_tokens
//! script:                                # optional; omit for live runs
//!   - tool_calls:
//!       - { name: get_weather, arguments: { city: Springfield } }
//!       # Arguments may also be a JSON string, as a provider sends them:
//!       # - { name: get_weather, arguments: '{"city":"Springfield"}' }
//!   - text: It is sunny and 21C in Springfield.
//! expect:
//!   status: completed
//!   final_text_contains: [sunny]
//!   tools_in_order: [get_weather]
//!   tool_call_counts: { get_weather: 1 }
//!   receipt_statuses: { get_weather: [succeeded] }
//!   no_failed_receipts: true
//!   proposed:                            # same assertions, on model proposals
//!     tool_call_counts: { get_weather: 1 }
//! ```
//!
//! Unknown fields are rejected everywhere, so a typo fails loudly instead
//! of silently dropping an assertion. A [`Subject`] can add its own typed
//! `setup:` block and `expect.ext:` block; load those with
//! [`Scenario::parse_yaml`] or [`Suite::load_dir`], which name the types.
//!
//! ## Running a suite against a live provider
//!
//! ```no_run
//! use std::sync::Arc;
//!
//! use aidaemon_core::{Agent, Provider};
//! use aidaemon_eval::{compare, FnFactory, Report, Scenario, Suite};
//!
//! # fn my_live_provider() -> Arc<dyn Provider> { unimplemented!() }
//! # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! // Your real provider (for example from the `aidaemon-providers` crate).
//! let live: Arc<dyn Provider> = my_live_provider();
//!
//! let factory = FnFactory::new(|scenario: &Scenario, provider: Arc<dyn Provider>| {
//!     let builder = Agent::builder("my-agent").allow_unguarded()
//!         .model("my-model")
//!         .provider(provider);
//!     // Add your tools and hooks here, then apply the scenario's
//!     // instructions and limits.
//!     Ok(scenario.configure(builder).build()?)
//! })
//! .with_live_provider(live);
//!
//! let suite = Suite::from_dir("evals/")?.parallelism(2);
//! let report = suite.run(&factory, 10).await; // 10 runs per scenario
//! println!("{}", report.to_markdown());
//!
//! // Compare against a saved baseline.
//! let baseline = Report::from_json(&std::fs::read_to_string("baseline.json")?)?;
//! println!("{}", compare(&baseline, &report).to_markdown());
//! std::fs::write("candidate.json", report.to_json())?;
//! # Ok(())
//! # }
//! ```

mod report;
mod runner;
mod scenario;
mod suite;

#[cfg(feature = "examples")]
pub mod examples;

pub use report::{compare, Comparison, ComparisonRow, FailureCount, Report, ScenarioReport};
pub use runner::{
    check_expectations, check_trajectory, AgentFactory, AgentSubject, CallRecord, CaseResult,
    FactoryError, FnFactory, Subject, Trajectory,
};
pub use scenario::{
    status_name, EvalError, Expect, ExpectedStatus, NoExtension, Scenario, ScenarioLimits,
    ScriptedArguments, ScriptedCall, ScriptedReply, ScriptedUsage, ToolLaneExpect,
};
pub use suite::Suite;
