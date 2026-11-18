//! The daemon's turn on each context budgeter, against a live model (PLAN
//! "Waiting on the owner" item 5: merge the two only on a measured gain).
//!
//! ```text
//! ANTHROPIC_API_KEY=... AIDAEMON_MODEL=claude-haiku-5-5 EVAL_N=5 \
//!   cargo run -p aidaemon-eval --features examples --example live_budgeters -- <out_dir> [arm ...]
//! ```
//! `EVAL_PROVIDER=openrouter` runs on OpenRouter with `PROVIDER_API_KEY`
//! (default model `moonshotai/kimi-k2.5`) instead.
//!
//! Arms:
//! - `daemon`: what the daemon's model call does
//!   (`aidaemon_runtime::engine::turn_engine::model_call`): each tool result
//!   capped at the configured character limit (`compress_tool_result`), then
//!   the one request budget, `engine::provider::fit_request` (the policies
//!   crate's `ContextWindow` over the prefix, the tool definitions and the
//!   run) sized by `context_budget::request_window` (the model's window less
//!   the response reserve, scaled by the per-model token calibration the
//!   provider's reported usage feeds). A provider's context-overflow error
//!   records the window it stated and retries, as the daemon does.
//! - `context_window`: the same result cap, then the policies crate's plain
//!   `ContextWindow` with the window less the response reserve, no
//!   calibration, no prefix.
//!
//! The daemon's earlier budgeter (role-quota fitting, a model-written
//! summary at 80% of the budget) was replaced by `fit_request` after the
//! 2c753f97 run; its numbers are printed beside the `daemon` arm as
//! [`BASELINE_2C753F97`].
//!
//! Both see the model as one with the window a scenario's
//! `context_cap:<tokens>` tag gives it, the budget the daemon would compute
//! for such a model. Scenarios default to the two that stress context
//! (`context_trail`, `context_tally`); `EVAL_ONLY=name,name` picks others.
//! Each arm writes `<out_dir>/<arm>.json` and prints its report and its
//! token split; the `daemon` arm also prints itself beside the 2c753f97
//! baseline. Token calibration is per process, as in the daemon: the
//! `daemon` arm's runs share it.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use aidaemon_core::{
    Hook, HookContext, ModelDecision, ModelErrorDecision, ModelRequest, ModelResponse, Provider,
    ProviderError, StreamEvent, Usage,
};
use aidaemon_eval::examples::{ExampleExpect, ExampleFactory, ExampleSubject};
use aidaemon_eval::{NoExtension, Report, Suite, compare};
use aidaemon_policies::ContextWindow;
use aidaemon_providers::Anthropic;
use aidaemon_runtime::context_budget::{
    calibrated_tokens, compress_tool_result, record_token_estimate_calibration, request_window,
};
use aidaemon_runtime::engine::provider::{
    RunMessage, fit_request, to_daemon_messages_from, values_to_sdk,
};
use aidaemon_runtime::engine::turn_engine::model_call::MAX_CONTEXT_OVERFLOW_RETRIES;
use aidaemon_runtime::settings::ContextWindowConfig;
use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;

const ARMS: &[&str] = &["daemon", "context_window"];
const DEFAULT_SCENARIOS: &str = "context_trail,context_tally";

/// The daemon's earlier budgeter on Haiku (claude-haiku-5-5), N=5, a
/// 10k-token window, at 2c753f97: (scenario, passes of 5, mean turns, mean
/// billable tokens per run). context_trail hit the 20-turn limit every run;
/// context_tally sent ~11.6k tokens against the 10k window and failed on
/// the provider every run. Spend over the 10 runs: 1.31M input, 20.6k output.
const BASELINE_2C753F97: &[(&str, usize, f64, f64)] = &[
    ("context_trail", 0, 20.0, 190_554.0),
    ("context_tally", 0, 2.0, 921.8),
];

/// Every call's usage.
#[derive(Default)]
struct Spend {
    input: AtomicU64,
    cached: AtomicU64,
    created: AtomicU64,
    output: AtomicU64,
    calls: AtomicU64,
}

impl Spend {
    fn add(&self, usage: &Usage) {
        self.input.fetch_add(usage.input_tokens, Ordering::Relaxed);
        self.cached
            .fetch_add(usage.cached_input_tokens, Ordering::Relaxed);
        self.created
            .fetch_add(usage.cache_creation_tokens, Ordering::Relaxed);
        self.output
            .fetch_add(usage.output_tokens, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
    }

    fn take(&self) -> [u64; 5] {
        [
            &self.input,
            &self.cached,
            &self.created,
            &self.output,
            &self.calls,
        ]
        .map(|counter| counter.swap(0, Ordering::Relaxed))
    }
}

/// The live provider, counting what every call used.
struct Counted {
    inner: Arc<dyn Provider>,
    spend: Arc<Spend>,
}

#[async_trait]
impl Provider for Counted {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ProviderError> {
        let response = self.inner.complete(request).await?;
        self.spend.add(&response.usage);
        Ok(response)
    }

    fn stream<'a>(
        &'a self,
        request: &'a ModelRequest,
    ) -> BoxStream<'a, Result<StreamEvent, ProviderError>> {
        let spend = self.spend.clone();
        Box::pin(self.inner.stream(request).inspect(move |event| {
            if let Ok(StreamEvent::Done(response)) = event {
                spend.add(&response.usage);
            }
        }))
    }

    fn unavailable_until(&self, model: &str) -> Option<std::time::SystemTime> {
        self.inner.unavailable_until(model)
    }
}

/// The daemon's configuration for a model with a `window`-token context:
/// its budget and its per-result character cap as a config would set them.
fn daemon_config(model: &str, window: usize) -> ContextWindowConfig {
    let mut config = ContextWindowConfig::default();
    config.model_budgets.insert(model.to_string(), window);
    // The production default cap (16k characters). Scaled to a 10k window it
    // would be 1.25k and no request would ever reach a budgeter; at 16k the
    // results reach the model whole, as on a production window.
    config
        .model_tool_result_chars
        .insert(model.to_string(), config.max_tool_result_chars);
    config
}

/// Each tool result as the daemon's tool dispatch leaves it: over the cap,
/// head and tail with a truncation notice.
fn cap_results(request: &mut ModelRequest, config: &ContextWindowConfig) {
    let max_chars = config.tool_result_chars_for(&request.model);
    for message in &mut request.messages {
        if message.tool_call_id.is_some() {
            let name = "tool";
            message.content = compress_tool_result(name, &message.content, max_chars);
        }
    }
}

/// The daemon's budgeting of one model call, as
/// `turn_engine::model_call` does it: the starting transcript is the wire
/// prefix, the rest the run.
struct DaemonBudget {
    config: ContextWindowConfig,
    /// The starting transcript's length (the daemon's wire prefix).
    prefix_len: Mutex<Option<usize>>,
    /// The last request's model and raw estimate, for calibration.
    sent: Mutex<Option<(String, usize)>>,
    overflow_retries: AtomicU32,
}

#[async_trait]
impl Hook for DaemonBudget {
    fn name(&self) -> &'static str {
        "daemon_budget"
    }

    async fn before_model(&self, _ctx: &HookContext<'_>, request: &mut ModelRequest) {
        cap_results(request, &self.config);
        let prefix_len = *self
            .prefix_len
            .lock()
            .unwrap()
            .get_or_insert(request.messages.len());
        let split = prefix_len.min(request.messages.len());
        let prefix = to_daemon_messages_from(&request.messages[..split]);
        let run = request.messages[split..]
            .iter()
            .cloned()
            .map(RunMessage::Kernel)
            .collect();
        let window = request_window(&request.model, &self.config);
        let fitted = fit_request(&prefix, run, &request.tools, &window);
        *self.sent.lock().unwrap() = Some((request.model.clone(), fitted.estimated_tokens));
        request.messages = values_to_sdk(&fitted.messages);
    }

    async fn after_model(
        &self,
        _ctx: &HookContext<'_>,
        response: &mut ModelResponse,
    ) -> ModelDecision {
        if let Some((model, estimate)) = self.sent.lock().unwrap().take() {
            record_token_estimate_calibration(
                &model,
                estimate,
                response.usage.input_tokens as usize,
            );
        }
        ModelDecision::Accept
    }

    async fn on_model_error(
        &self,
        _ctx: &HookContext<'_>,
        error: &ProviderError,
    ) -> ModelErrorDecision {
        let sent = self.sent.lock().unwrap().take();
        if let (ProviderError::ContextOverflow { limit, .. }, Some((model, estimate))) =
            (error, sent)
            && self.overflow_retries.fetch_add(1, Ordering::Relaxed) < MAX_CONTEXT_OVERFLOW_RETRIES
        {
            aidaemon_runtime::settings::record_context_overflow(
                &model,
                limit.map(|l| l as usize),
                calibrated_tokens(&model, estimate),
            );
            return ModelErrorDecision::Retry {
                after: std::time::Duration::ZERO,
            };
        }
        ModelErrorDecision::Fail
    }
}

/// The result cap, then `ContextWindow` with the daemon's budget.
struct CapThenWindow {
    config: ContextWindowConfig,
    window: ContextWindow,
}

#[async_trait]
impl Hook for CapThenWindow {
    fn name(&self) -> &'static str {
        "cap_then_context_window"
    }

    async fn before_model(&self, ctx: &HookContext<'_>, request: &mut ModelRequest) {
        cap_results(request, &self.config);
        self.window.before_model(ctx, request).await;
    }
}

/// This run's `daemon` arm beside the earlier budgeter's (2c753f97).
fn baseline_table(report: &Report) -> String {
    let mut table = String::from(
        "| scenario | 2c753f97 passes | now passes | 2c753f97 turns | now turns | \
         2c753f97 billable/run | now billable/run |\n|---|---|---|---|---|---|---|\n",
    );
    for (name, passes, turns, billable) in BASELINE_2C753F97 {
        let Some(now) = report.scenarios.iter().find(|s| s.name == *name) else {
            continue;
        };
        table.push_str(&format!(
            "| {name} | {passes}/5 | {}/{} | {turns:.1} | {:.1} | {billable:.0} | {:.0} |\n",
            now.passes, now.runs, now.mean_turns, now.mean_billable_tokens
        ));
    }
    table
}

fn window_of(scenario: &aidaemon_eval::Scenario) -> usize {
    scenario
        .tags
        .iter()
        .filter_map(|t| t.strip_prefix("context_cap:"))
        .find_map(|n| n.parse().ok())
        .unwrap_or(200_000)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let out = PathBuf::from(
        args.next()
            .ok_or("usage: live_budgeters <out_dir> [arm ...]")?,
    );
    let mut arms: Vec<String> = args.collect();
    if arms.is_empty() {
        arms = ARMS.iter().map(|a| a.to_string()).collect();
    }
    if let Some(bad) = arms.iter().find(|a| !ARMS.contains(&a.as_str())) {
        return Err(format!("unknown arm `{bad}`; one of {ARMS:?}").into());
    }
    std::fs::create_dir_all(&out)?;
    let n: usize = std::env::var("EVAL_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    // Each key only with its own provider: Anthropic with
    // `ANTHROPIC_API_KEY`, or OpenRouter (`EVAL_PROVIDER=openrouter`) with the
    // key the daemon's config gives its OpenRouter fallback.
    let (live, model): (Arc<dyn Provider>, String) = match std::env::var("EVAL_PROVIDER").as_deref()
    {
        Ok("openrouter") => {
            let key = std::env::var("PROVIDER_API_KEY")
                .map_err(|_| "set PROVIDER_API_KEY (the OpenRouter key)")?;
            let provider = aidaemon_providers::OpenAiCompatible::builder()
                .name("openrouter")
                .base_url("https://openrouter.ai/api/v1")
                .api_key(key)
                .max_tokens(4096)
                .build()?;
            let model =
                std::env::var("AIDAEMON_MODEL").unwrap_or_else(|_| "moonshotai/kimi-k2.5".into());
            (Arc::new(provider), model)
        }
        _ => {
            let key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| "set ANTHROPIC_API_KEY")?;
            let model =
                std::env::var("AIDAEMON_MODEL").unwrap_or_else(|_| "claude-haiku-5-5".into());
            (Arc::new(Anthropic::builder().api_key(key).build()?), model)
        }
    };
    let spend = Arc::new(Spend::default());
    let provider: Arc<dyn Provider> = Arc::new(Counted {
        inner: live,
        spend: spend.clone(),
    });
    let mut suite: Suite<NoExtension, ExampleExpect> =
        Suite::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/scenarios"))?
            .live()
            .parallelism(4);
    let only = std::env::var("EVAL_ONLY").unwrap_or_else(|_| DEFAULT_SCENARIOS.into());
    let names: Vec<&str> = only.split(',').map(str::trim).collect();
    suite.scenarios.retain(|s| names.contains(&s.name.as_str()));
    if suite.scenarios.is_empty() {
        return Err(format!("EVAL_ONLY matched no scenario: {only}").into());
    }
    let runs = (n * suite.scenarios.len()) as f64;

    for arm in arms {
        let hook_arm = arm.clone();
        let hook_model = model.clone();
        let factory = ExampleFactory::with_live_provider(provider.clone())
            .model(model.clone())
            .with_hooks(move |scenario| {
                let window = window_of(scenario);
                let config = daemon_config(&hook_model, window);
                let hook: Arc<dyn Hook> = match hook_arm.as_str() {
                    "daemon" => Arc::new(DaemonBudget {
                        config,
                        prefix_len: Mutex::new(None),
                        sent: Mutex::new(None),
                        overflow_retries: AtomicU32::new(0),
                    }),
                    _ => Arc::new(CapThenWindow {
                        window: ContextWindow::new(window.saturating_sub(
                            aidaemon_runtime::context_budget::CONTEXT_RESPONSE_RESERVE_TOKENS,
                        )),
                        config,
                    }),
                };
                vec![hook]
            });
        let report = suite.run_subject(&ExampleSubject::new(&factory), n).await;
        let [input, cached, created, output, calls] = spend.take();
        println!("## {arm} ({model}, N={n})\n\n{}", report.to_markdown());
        println!(
            "Spend over {runs} runs: {calls} calls, input {input} \
             (cache read {cached}, cache write {created}), output {output}; \
             billable/run {:.0}\n",
            (input.saturating_sub(cached) + output) as f64 / runs
        );
        std::fs::write(out.join(format!("{arm}.json")), report.to_json())?;
        std::fs::write(
            out.join(format!("{arm}.spend.json")),
            format!(
                "{{\"input\":{input},\"cache_read\":{cached},\"cache_write\":{created},\
                 \"output\":{output},\"calls\":{calls},\"runs\":{runs}}}"
            ),
        )?;
        if arm == "daemon" {
            println!("{}", baseline_table(&report));
        }
        let daemon = out.join("daemon.json");
        if arm != "daemon" && daemon.exists() {
            let base = Report::from_json(&std::fs::read_to_string(daemon)?)?;
            println!("{}", compare(&base, &report).to_markdown());
        }
    }
    Ok(())
}
