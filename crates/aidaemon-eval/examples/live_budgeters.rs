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
//! - `daemon`: what the daemon's model call does today
//!   (`aidaemon_runtime::engine::turn_engine`): each tool result capped at
//!   the configured character limit (`compress_tool_result`), the run's
//!   older rounds replaced by a model-written summary once the request
//!   passes 80% of the budget, then role-quota fitting
//!   (`fit_run_values`) as the fallback, over the wire-format JSON.
//! - `context_window`: the same result cap, then the policies crate's
//!   `ContextWindow` over the typed messages, with the daemon's own budget
//!   (the window less the response reserve).
//!
//! Both see the model as one with the window a scenario's
//! `context_cap:<tokens>` tag gives it, the budget the daemon would compute
//! for such a model. Scenarios default to the two that stress context
//! (`context_trail`, `context_tally`); `EVAL_ONLY=name,name` picks others.
//! Each arm writes `<out_dir>/<arm>.json` and prints its report and its
//! token split (summary calls included, which the report's per-run
//! billable tokens leave out).

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use aidaemon_core::{
    Hook, HookContext, ModelRequest, ModelResponse, Provider, ProviderError, StreamEvent, Usage,
};
use aidaemon_eval::examples::{ExampleExpect, ExampleFactory, ExampleSubject};
use aidaemon_eval::{NoExtension, Report, Suite, compare};
use aidaemon_policies::ContextWindow;
use aidaemon_providers::Anthropic;
use aidaemon_runtime::context_budget::{
    compress_tool_result, compute_available_budget_precomputed, estimate_multimodal_message_tokens,
};
use aidaemon_runtime::engine::compaction::{
    COMPACT_AT_PERCENT, KEEP_RECENT_PERCENT, RunCompaction, SUMMARY_MAX_OUTPUT_TOKENS,
    compacted_view, compaction_cut, summary_request,
};
use aidaemon_runtime::engine::provider::{
    fit_run_values, to_daemon_messages_from, to_daemon_tools, values_to_sdk,
};
use aidaemon_runtime::settings::ContextWindowConfig;
use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;

const ARMS: &[&str] = &["daemon", "context_window"];
const DEFAULT_SCENARIOS: &str = "context_trail,context_tally";

/// Every call's usage, summary calls included.
#[derive(Default)]
struct Spend {
    input: AtomicU64,
    cached: AtomicU64,
    created: AtomicU64,
    output: AtomicU64,
    calls: AtomicU64,
    summary_calls: AtomicU64,
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

    fn take(&self) -> [u64; 6] {
        [
            &self.input,
            &self.cached,
            &self.created,
            &self.output,
            &self.calls,
            &self.summary_calls,
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
/// `turn_engine::model_call` and `turn_engine::compaction` do it.
struct DaemonBudget {
    config: ContextWindowConfig,
    summarizer: Arc<dyn Provider>,
    spend: Arc<Spend>,
    /// The starting transcript's length (the daemon's wire prefix).
    prefix_len: Mutex<Option<usize>>,
    compaction: Mutex<Option<RunCompaction>>,
}

impl DaemonBudget {
    async fn compacted_run(
        &self,
        model: &str,
        prefix_tokens: usize,
        run: &[serde_json::Value],
        budget: usize,
    ) -> Vec<serde_json::Value> {
        let current = self.compaction.lock().unwrap().clone();
        let view = compacted_view(run, current.as_ref());
        let view_tokens = estimate_multimodal_message_tokens(&view);
        if prefix_tokens + view_tokens <= budget * COMPACT_AT_PERCENT / 100 {
            return view;
        }
        let covered = current.as_ref().map_or(0, |c| c.covered);
        let keep_tokens = budget.saturating_sub(prefix_tokens) * KEEP_RECENT_PERCENT / 100;
        let Some(cut) = compaction_cut(&run[covered..], keep_tokens).map(|cut| covered + cut)
        else {
            return view;
        };
        if cut <= covered {
            return view;
        }
        let request = summary_request(
            current.as_ref().map(|c| c.summary.as_str()),
            &run[covered..cut],
        );
        let request = ModelRequest {
            model: model.to_string(),
            messages: values_to_sdk(&request),
            tools: Vec::new(),
            tool_choice: aidaemon_core::ToolChoice::None,
            max_output_tokens: Some(SUMMARY_MAX_OUTPUT_TOKENS),
            options: Default::default(),
        };
        self.spend.summary_calls.fetch_add(1, Ordering::Relaxed);
        let summary = match self.summarizer.complete(&request).await {
            Ok(response) => response.text,
            Err(error) => {
                eprintln!("summary call failed, fitting by quota: {error}");
                return view;
            }
        };
        if summary.trim().is_empty() {
            return view;
        }
        let compaction = RunCompaction {
            covered: cut,
            summary: summary.trim().to_string(),
        };
        let view = compacted_view(run, Some(&compaction));
        *self.compaction.lock().unwrap() = Some(compaction);
        view
    }
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
        let tools = to_daemon_tools(&request.tools);
        let budget = compute_available_budget_precomputed(&request.model, 0, &tools, &self.config);
        let split = prefix_len.min(request.messages.len());
        let prefix = to_daemon_messages_from(&request.messages[..split]);
        let run = to_daemon_messages_from(&request.messages[split..]);
        let prefix_tokens = estimate_multimodal_message_tokens(&prefix);
        let run = self
            .compacted_run(&request.model, prefix_tokens, &run, budget)
            .await;
        request.messages = values_to_sdk(&fit_run_values(&prefix, run, budget));
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
        let summarizer = provider.clone();
        let hook_spend = spend.clone();
        let factory = ExampleFactory::with_live_provider(provider.clone())
            .model(model.clone())
            .with_hooks(move |scenario| {
                let window = window_of(scenario);
                let config = daemon_config(&hook_model, window);
                let hook: Arc<dyn Hook> = match hook_arm.as_str() {
                    "daemon" => Arc::new(DaemonBudget {
                        config,
                        summarizer: summarizer.clone(),
                        spend: hook_spend.clone(),
                        prefix_len: Mutex::new(None),
                        compaction: Mutex::new(None),
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
        let [input, cached, created, output, calls, summaries] = spend.take();
        println!("## {arm} ({model}, N={n})\n\n{}", report.to_markdown());
        println!(
            "Spend over {runs} runs: {calls} calls ({summaries} summaries), input {input} \
             (cache read {cached}, cache write {created}), output {output}; \
             billable/run {:.0}\n",
            (input.saturating_sub(cached) + output) as f64 / runs
        );
        std::fs::write(out.join(format!("{arm}.json")), report.to_json())?;
        std::fs::write(
            out.join(format!("{arm}.spend.json")),
            format!(
                "{{\"input\":{input},\"cache_read\":{cached},\"cache_write\":{created},\
                 \"output\":{output},\"calls\":{calls},\"summary_calls\":{summaries},\"runs\":{runs}}}"
            ),
        )?;
        let baseline = out.join("daemon.json");
        if arm != "daemon" && baseline.exists() {
            let base = Report::from_json(&std::fs::read_to_string(baseline)?)?;
            println!("{}", compare(&base, &report).to_markdown());
        }
    }
    Ok(())
}
