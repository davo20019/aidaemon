# Reliability and telemetry for auxiliary LLM calls (state as of November 2026)

## 1. Separate auxiliary calls versus routing everything through the main agent loop

### Takeaway
I found no published consensus and no source that argues for putting every auxiliary call through the loop. Shipped agents use a spectrum (plain one-shot call, cache-sharing fork, subagent, loop step), and they pick by whether the call needs tools, the conversation, or independence from the agent's reasoning.

### Cited Findings
**What Claude Code does (primary docs)**
- Its telemetry names three classes of request. `query_source` on metrics is `main`, `subagent` or `auxiliary`, and event-level values identify the subsystem (for example `compact`) — [Claude Code monitoring](https://code.claude.com/docs/en/monitoring-usage).
- Compaction is "a separate request" that repeats the conversation's system prompt, tools and history and reads the warm cache. — [Claude Code prompt caching](https://code.claude.com/docs/en/prompt-caching).
- Prompt suggestions are a short request to the session's model after each response. It reuses the conversation cache, so it is mostly cache reads, and it is skipped when the account is near its usage limit — [Claude Code costs](https://code.claude.com/docs/en/costs).
- A subagent has its own prefix and does not read the parent's cache on its first request. A fork inherits the prefix exactly and does — [Claude Code prompt caching](https://code.claude.com/docs/en/prompt-caching).
- The auto-mode safety classifier is a separate gate outside the agent's context. It sees user messages and tool calls only; assistant text and tool outputs are stripped so the agent cannot argue its case and injected content cannot reach it. A variant with tool calls only lost the user-intent anchor and over-blocked — [Anthropic engineering, auto mode](https://www.anthropic.com/engineering/claude-code-auto-mode).

**Other agents**
- Codex auto-review routes approval requests "to a separate reviewer agent". It can run read-only checks (for example, inspecting a directory before approving its deletion) and sees a compact transcript plus the exact request, without hidden reasoning — [Codex auto-review docs](https://learn.chatgpt.com/docs/sandboxing/auto-review). A mirror of the original commit says the reviewer "runs as a normal subagent" reusing subagent machinery — [third-party mirror of codex-rs commit #13692](https://forge.lthn.ai/core/core-agent-ide/commit/e84ee33cc02e693a3cf66204c72cb37e8dda3ed6). This is the main counter-example to tool-less judges.
- Codex configures memory extraction and consolidation through separate `extract_model` and `consolidation_model` keys. A third-party report describes surprise requests to unselected models when they are unset under a custom provider — [third-party write-up](https://aident.ai/blog/fix-codex-memory-unselected-models).
- Hermes Agent resolves vision, compression, MCP, approval, title generation, review and triage separately from the main loop, each with its own provider chain — [Hermes fallback docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/fallback-providers).
- OpenAI Agents SDK guardrails can run a smaller, cheaper model or call an external classifier directly. They run in parallel by default, which means the agent may already have spent tokens and run tools when the tripwire fires; `run_in_parallel=False` avoids that — [Agents SDK guardrails](https://openai.github.io/openai-agents-python/guardrails/).
- Vercel's AI SDK now has a typed decision primitive for Choice, Score and Boolean questions with per-question confidence. That is a typed one-shot call as a product category — [Vercel decision fallbacks](https://vercel.com/docs/ai-gateway/models-and-providers/decision-fallbacks).

**Durability**
- Temporal's integrations run model calls as activities so they are durable and retryable — [Temporal blog, a vendor source](https://temporal.io/blog/introducing-temporal-and-agentic-sandboxes-openai-agents-sdk). A third-party guide notes Schedule-to-Close covers all retries while Start-to-Close covers one attempt — [third-party summary](https://dev.to/jangwook_kim_e31e7291ad98/temporal-for-ai-agents-durable-execution-guide-2026-9hf).
- OpenAI Agents SDK fails closed on retry for stateful requests (`previous_response_id`, `conversation_id`) when replay safety is unknown, and never retries a stream after output has started — [Agents SDK models](https://openai.github.io/openai-agents-python/models/).

**Cost**
- Delegation pays for independent work or work larger than one context. It did not pay for single dependent chains that fit in one context — [Anthropic, optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence). This evidence concerns worker agents, not one-shot side calls.

### Inferences
- A useful sort is by the need the call has. No tools and no conversation: plain one-shot. Full conversation needed: same-model fork, for the cache. Must search or read: a subagent or loop step. Must be independent of the agent's reasoning: a call that is deliberately not in the loop's context.
- "Most of these should never join the agent loop" is supported for judges, routers and memory extraction. Running a safety judge in the loop would undo the input-stripping that Anthropic presents as the point of its design.
- "Everything through the loop" gains transcript storage, tracing and authority hooks for free, but those are properties of an instrumented path, and a one-shot helper can have them too. Its costs are prompt inheritance, tool exposure and a harder-to-reason-about loop budget.
- Durability is only needed where losing the result matters. A router or judge verdict is cheap to recompute, while a long memory consolidation may deserve a checkpoint. A judge verdict should be persisted as a receipt for audit rather than for replay.

### Gaps
- No controlled comparison of side call versus loop step on cost, latency or quality was found.
- I could not read current codex-rs guardian source. The subagent and fail-closed details come from a mirrored commit and third-party blogs ([example](https://codex.danielvaughan.com/2026/04/20/when-guardian-approval-goes-wrong-failure-modes-escalation-patterns/)).
- Most page extractions throughout these notes went through a summarizing fetch tool. Exact defaults should be re-checked against the page before being copied into code.

## 2. How gateways and SDKs structure retries, fallbacks, circuit breakers, deadlines

### Takeaway
Mature systems retry first at one layer, then fail over by error class, and they keep quota, auth and bad-request errors out of the retry path. They differ sharply on refusals, on what an open circuit means, and on whether fallback is built in at all.

### Cited Findings
**Gateways**
- LiteLLM retries within a group first, then falls back in order. It has separate chains for content-policy errors, context-window errors and everything else — [LiteLLM reliability](https://docs.litellm.ai/docs/proxy/reliability).
- LiteLLM defaults: `cooldown_time` 5 seconds; without `allowed_fails`, immediate cooldown on a 429, or on 401/404/408, or above 50% failures in the minute. `RetryPolicy` and `AllowedFailsPolicy` are per error class. Rate-limit errors back off exponentially and generic errors retry immediately. `max_retries` is pinned to 0 on router calls so only one retry loop exists. A concurrency cap returns an immediate 429 with no queue — [LiteLLM routing](https://docs.litellm.ai/docs/routing).
- OpenRouter's `models` array falls back on context-length errors, moderation flags, rate limits and downtime. The served model is reported in the response, and billing follows the serving model — [OpenRouter fallbacks](https://openrouter.ai/docs/guides/routing/model-fallbacks).
- OpenRouter's default routing weighs reliability and 429s continuously. It has no documented fixed cooldown or timeout — [OpenRouter provider routing](https://openrouter.ai/docs/guides/routing/provider-selection).
- Vercel AI Gateway tries each provider for a model, then the next model, and returns `modelAttempts` with per-attempt status, timing and credential type — [Vercel model fallbacks](https://vercel.com/docs/ai-gateway/models-and-providers/model-fallbacks).
- Vercel separates execution-failure fallback from a successful-but-uncertain rerun. A triggered rerun bills both stages, runs them sequentially, and stops at two stages — [Vercel decision fallbacks](https://vercel.com/docs/ai-gateway/models-and-providers/decision-fallbacks).
- Portkey retries 429, 500, 502, 503, 504 and 529, up to 5 attempts at 1, 2, 4, 8 and 16 seconds. Cumulative `Retry-After` waiting is capped at 60 seconds, and a single larger value fails immediately — [Portkey retries](https://portkey.ai/docs/product/ai-gateway/automatic-retries).
- Portkey fallback triggers on any non-2xx by default — [Portkey fallbacks](https://portkey.ai/docs/product/ai-gateway/fallbacks).
- Portkey circuit breaker: threshold or percentage, cooldown of at least 30 seconds, tracked per strategy path. **If every target is open the breaker is bypassed and requests go to all targets** — [Portkey circuit breaker](https://portkey.ai/docs/product/ai-gateway/circuit-breaker).

**SDKs and frameworks**
- LangChain `ModelRetryMiddleware`: 3 total attempts, backoff factor 2, 1 second initial, 60 second cap, ±25% jitter. **`on_failure` defaults to `continue`, which returns an AIMessage with error details rather than raising.** It does not retry refusals, but `ModelFallbackMiddleware` will try the next model on a refusal when placed before refusal handling — [LangChain middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in).
- OpenAI Agents SDK: no retries unless opted in via `ModelRetrySettings`, and each retry gets a fresh `ModelSettings.timeout`. I found no documented cross-model failover — [Agents SDK models](https://openai.github.io/openai-agents-python/models/).
- Google ADK: official docs address 429 with higher quota and client-side `retry_options` ([ADK Gemini docs](https://adk.dev/agents/models/google-gemini/)). I found no official model fallback; a plugin is a community proposal ([adk-python-community #88](https://github.com/google/adk-python-community/issues/88)).

**Agent CLIs**
- Claude Code fallback chain: capped at three models, triggered by overload, unavailability or non-retryable server errors. **Authentication, billing, rate-limit, request-size, transport and policy denials never trigger a switch.** It lasts one turn. Compaction will not fall back to a smaller context window. The chain is not shown at startup or in `/status` — [Claude Code model config](https://code.claude.com/docs/en/model-config).
- Hermes: main fallback triggers on 429/500/502/503 after retries and on 401/403/404 immediately, plus repeated malformed or empty 200 responses. It activates at most once per turn, with a 60 second to 4 hour cooldown when no reset time is known. Auxiliary tasks walk their ladder on capacity errors (402, daily-quota 429, connection failure), and **a transient 429 with `Retry-After` does not trigger the ladder** — [Hermes fallback docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/fallback-providers).
- Codex `request_max_retries` defaults to 4 and `stream_max_retries` to 5, per older config docs and mirrors. Treat as possibly stale — [third-party mirror of docs/config.md](https://fossies.org/linux/codex-rust/docs/config.md).

**Quota versus transient errors**
- Anthropic: rate limits are per organization and per model class (some classes share a pool). A monthly spend-cap 429 has **no `retry-after`, uses the same `rate_limit_error` type, and is told apart by `error.details.error_code: enforced_spend_limit_reached`**. A user-set spend limit returns 400 — [Anthropic rate limits](https://platform.claude.com/docs/en/api/rate-limits).
- Anthropic: 529 is overload. Official SDKs retry twice by default and honor `retry-after` — [Anthropic errors](https://platform.claude.com/docs/en/api/errors).
- OpenAI: limits are per organization and project, with some models sharing a pool. **Failed requests count toward the limit.** Treat `Retry-After` as a minimum and add jitter, cap attempts and total retry time, disable SDK retries if you run your own loop, defer rather than retry early, and never retry quota or billing errors. 429 `slow_down` and 503 `server_is_overloaded` are distinct — [OpenAI rate limits](https://developers.openai.com/api/docs/guides/rate-limits).
- Gemini: limits are per project and vary by model, RPD resets at midnight Pacific, and a spend-based limit on a rolling 10-minute window returns 429 — [Gemini rate limits](https://aistudio.google.com/docs/rate-limits).
- Claude Code documents the same split for subscriptions: a session or weekly limit is shared across models, so switching does not help, while a model-specific limit can be escaped by switching family — [Claude Code costs](https://code.claude.com/docs/en/costs).

**Retry storms, deadlines, hedging (2016 Google SRE material, still the reference)**
- Retry budgets: 3 attempts per request and 10% of a client's requests. Worst-case growth falls from nearly 3x to about 1.1x. Only the layer directly above the rejecting layer retries, and client-side throttling begins when requests reach 2x accepts — [SRE, handling overload](https://sre.google/sre-book/handling-overload/).
- Three layers each making 4 attempts yields 64 attempts on the backend. Propagate one absolute deadline, subtract a margin, and check remaining time before each stage. Do not retry permanent errors — [SRE, cascading failures](https://sre.google/sre-book/addressing-cascading-failures/).
- Hedging: I found no gateway or SDK above that documents built-in request hedging. Evidence on LLM hedging is practitioner blogs and library READMEs ([third-party example](https://tianpan.co/blog/2026-05-02-tail-tolerant-retry-policy-llm-gateway-latency-cliff)). The discarded response still costs money, and SRE requires propagating cancellation to the redundant copy ([SRE](https://sre.google/sre-book/addressing-cascading-failures/)).

### Inferences
- Applied to the five aidaemon styles, as things to test rather than verdicts:
  - **Safety helper.** The closest in shape to Claude Code's documented policy (infrastructure-only failover, fail closed). Test that a 429 quota error is not classified as infrastructure, or the judge will walk the provider list on a spend cap.
  - **Router.** One attempt with a shared deadline matches a sheddable advisory call. Test that "gives up quietly" increments a counter.
  - **Background chain.** The widest walk. On an account-scoped quota error, trying sibling models on the same account is wasted load, while a per-model limit makes it useful. It needs a typed quota scope rather than a blind walk.
  - **RuntimeProvider.** 3 attempts × 2 models × 360 seconds is up to 2,160 seconds of wall clock before backoff, unless a shared deadline bounds it. If an SDK or transport layer also retries, multiply again.
  - **Raw calls.** The strongest-model call with no telemetry is the weakest point regardless of which proposal wins.
- Every system surveyed has one retry mechanism with declared per-call or per-error policy. Hermes (per-task chains), LiteLLM (per-error policy) and Agents SDK (merged settings) are examples. None forces one policy on all calls, and none scatters independent loops. So "one typed helper" and "policies differ on purpose" are compatible if the helper has an execution core and a required, explicit policy, with no default failure mode.
- The counter-argument is strongest on failure semantics, where judges and background jobs should differ. It is weakest on plumbing, because "adopt the right existing policy" does nothing for telemetry or for the raw calls unless the existing paths are all instrumented.
- A helper's main risk is that migration silently changes tier, deadline or failover scope. Two styles both serving "some memory jobs" suggests at least part of the current variation is historical rather than designed, but only a call-by-call review can show that.
- The error taxonomy should distinguish at least: transient rate limit (honor delay), account quota or spend (stop), per-model limit (switch model), overload (backoff, maybe switch), auth (stop), bad request (stop), timeout, and a 200 with refusal or truncation.

### Gaps
- Google ADK's Python support for fallback and whether `retry_options` works reliably is unclear. One user reports it did nothing for 429s ([discussion, anecdotal](https://github.com/google/adk-python/discussions/3812)).
- LiteLLM's docs show no default for `num_retries` on the pages fetched.
- No independent controlled hedging study against LLM APIs was found.

## 3. Background deadlines, concurrency, foreground/background isolation, prompt-caching effects

### Takeaway
Long attempt deadlines are defensible for deprioritized or known-long work, but vendor guidance does not extend that to small synchronous requests, and SRE warns against them generally. Quota protection comes from priority shedding and concurrency limits, and model or provider switches give up the cache in both directions.

### Cited Findings
- Deadlines far above mean latency exhaust resources. In SRE's example, 5% of requests hanging until a 100-second deadline turns a 5% error rate into about 80%. Checkpointing work is the listed exception — [SRE, cascading failures](https://sre.google/sre-book/addressing-cascading-failures/) (2016, server thread-pool context).
- OpenAI Flex: batch-rate pricing, a 10-minute default SDK timeout, examples at 15 minutes. 429 Resource Unavailable is not charged, and time-sensitive work should retry at standard tier — [OpenAI flex processing](https://developers.openai.com/api/docs/guides/flex-processing).
- Anthropic recommends streaming or batches for long requests, especially over 10 minutes, and warns networks can drop idle connections — [Anthropic errors](https://platform.claude.com/docs/en/api/errors).
- Batch tiers have separate limits at Anthropic ([rate limits](https://platform.claude.com/docs/en/api/rate-limits)) and Gemini ([rate limits](https://aistudio.google.com/docs/rate-limits)), which isolates them from interactive quota. But the Anthropic Batch API is not ZDR eligible because it stores jobs — [Anthropic data retention](https://platform.claude.com/docs/en/manage-claude/api-and-data-retention).
- Anthropic workspaces can cap a workload's RPM/ITPM/OTPM to protect others; organization limits still apply — [Anthropic rate limits](https://platform.claude.com/docs/en/api/rate-limits).
- SRE criticality classes: `CRITICAL_PLUS`, `CRITICAL`, `SHEDDABLE_PLUS`, `SHEDDABLE`. Batch defaults to `SHEDDABLE_PLUS` — [SRE, handling overload](https://sre.google/sre-book/handling-overload/). Drop background work first and cap any single client's share of capacity — [SRE, cascading failures](https://sre.google/sre-book/addressing-cascading-failures/).
- Claude Code skips prompt suggestions when near the usage limit — [Claude Code costs](https://code.claude.com/docs/en/costs).
- **Caching.** Each model has its own cache, so a switch recomputes the entire request; requests outside the main conversation generally get a 5-minute TTL, apart from a small server-controlled set of helper requests — [Claude Code prompt caching](https://code.claude.com/docs/en/prompt-caching).
- Claude Code holds workflow fan-out agents up to 5 seconds so they can read the first agent's cached prefix — [Claude Code prompt caching](https://code.claude.com/docs/en/prompt-caching).
- Hermes notes that switching provider and switching back each re-read history at full price, and offers `min_switch_reset_seconds` to skip a switch when the primary reopens soon — [Hermes fallback docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/fallback-providers).
- OpenRouter keeps follow-up requests on the same provider while the cache is warm — [OpenRouter provider routing](https://openrouter.ai/docs/guides/routing/provider-selection).
- Anthropic ITPM counts only uncached input, so cache reads do not consume it; an 80% hit rate at a 2M limit serves about 10M input tokens per minute — [Anthropic rate limits](https://platform.claude.com/docs/en/api/rate-limits).
- Changing `output_config.format` invalidates the prompt cache for that thread — [Anthropic structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).

### Inferences
- A 360-second attempt deadline suits a call that really streams long work. For a short summarization it mostly delays failure detection. A shared total deadline plus a stall or time-to-first-token check would separate "slow" from "hung". This is my reasoning, not a published LLM-specific result.
- Background work should take a separate concurrency limiter from foreground and shed when quota headroom is low. Rate-limit headers give the signal.
- Cache economics argue for two shapes: a small standalone prompt on a cheap model (the cache does not matter) or a same-model fork (the cache is the point). A cheap model fed a large conversation may give up the saving.
- Rate-limit interaction is provider-specific: on Anthropic, a warm-cache side call is nearly free against ITPM, while the same content on a different model counts fully against it.

### Gaps
- No published LLM-specific evidence on the right deadline distribution for short auxiliary calls.
- ChatGPT-subscription-backed access (relevant if a primary provider uses one) has no public limit semantics equivalent to the API docs that I found.

## 4. Cross-provider failover risks; should safety judges fail over differently?

### Takeaway
Risks are real and documented: unsupported parameters silently ignored, schema differences, cost, cache loss and different data terms. Practice fails judges closed. I found no shipped example of a safety judge failing over across vendors, and how refusals should be treated has no agreement.

### Cited Findings
**Behavior and schema**
- Portkey cautions that fallback models must be compatible with the use case and that latency and pricing differ — [Portkey fallbacks](https://portkey.ai/docs/product/ai-gateway/fallbacks).
- OpenAI's SDK docs warn some providers only support JSON output and not a specified schema, may return invalid JSON, and that LiteLLM and Any-LLM adapters are best effort. Validate structured outputs, tool calling and usage against the exact backend — [Agents SDK models](https://openai.github.io/openai-agents-python/models/).
- OpenRouter's `require_parameters` defaults to false, so a provider may receive a request and **ignore unsupported parameters**; response format acts as a soft preference. Set true to exclude them — [OpenRouter provider routing](https://openrouter.ai/docs/guides/routing/provider-selection).

**Data handling**
- OpenRouter defaults `data_collection` to `allow`. `zdr: true` restricts to ZDR endpoints, with Safety Retention endpoints still eligible unless settings say otherwise — [OpenRouter provider routing](https://openrouter.ai/docs/guides/routing/provider-selection).
- At Anthropic, Fable and Mythos models require 30-day retention. Even under ZDR, flagged content may be retained up to 2 years. On Bedrock and Google-managed Claude the cloud provider is the processor — [Anthropic data retention](https://platform.claude.com/docs/en/manage-claude/api-and-data-retention).

**Refusals: the field disagrees**
- LiteLLM has dedicated cross-provider `content_policy_fallbacks` ([docs](https://docs.litellm.ai/docs/proxy/reliability)). OpenRouter falls back on moderation flags by default ([docs](https://openrouter.ai/docs/guides/routing/model-fallbacks)). Vercel treats a refusal as a fallback trigger on decision requests ([docs](https://vercel.com/docs/ai-gateway/models-and-providers/decision-fallbacks)).
- Hermes treats a streamed refusal as terminal and does not retry ([docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/fallback-providers)). Claude Code blocks the action when its classifier refuses or returns unparseable output ([docs](https://code.claude.com/docs/en/permission-modes)).

**Judges**
- Claude Code auto mode denies on no verdict, early-ended or unreadable responses, or a failed check such as a timeout. After ten responses in a row with no verdict the turn can stop. Denials from a safety check refusing the classifier's request do not count toward the block thresholds — [Claude Code permission modes](https://code.claude.com/docs/en/permission-modes).
- If Sonnet 5 is unavailable the classifier falls back to the session's model, or an Opus model in a Fable session: same vendor, and a capability-preserving direction — [Claude Code permission modes](https://code.claude.com/docs/en/permission-modes).
- Codex surfaces timeouts separately from denials and tells the agent "a timeout alone is not proof that the action is unsafe" — [Codex auto-review](https://learn.chatgpt.com/docs/sandboxing/auto-review).
- OpenAI Agents SDK treats an output guardrail exception as an unknown verdict and raises — [Agents SDK guardrails](https://openai.github.io/openai-agents-python/guardrails/).
- Contrast: LangChain's retry middleware by default continues with an error message — [LangChain middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in). It is a general framework default, not a judge, but it shows why fail-open has to be designed against.

### Inferences
- Judges and background jobs should fail over differently, with opposite priorities. A judge should preserve meaning: fail closed, treat a timeout as "no verdict" rather than "deny for cause", and allow only candidates that passed that judge's own eval. A background job should preserve eventual completion: defer and retry later on the primary rather than exfiltrate content across more vendors to finish sooner.
- aidaemon's helper style has a risk worth testing: failing over to an unevaluated vendor can produce a confident ALLOW with a different calibration, which is worse than an honest no-verdict. Record the serving model in the receipt.
- Falling back on a refusal is policy shopping if the call is a safety judge. For background extraction it is merely wasteful.
- The background chain is the only style described that walks all failover providers for memory jobs, and memories are likely among the most private content the daemon holds. RuntimeProvider never crosses providers. Check whether that difference was chosen, and consider a per-call data-class rule.

### Gaps
- No public evaluation of how judge calibration shifts across vendors, and no example of a cross-vendor judge fallback in a shipped agent.
- Did not review OpenAI's and Google's retention pages in detail.

## 5. Model tier selection and cascades

### Takeaway
Vendor guidance is to start from an evaluation rather than a tier. Cheap models suit bounded, high-volume work, yet some side calls deliberately use a stronger or session model, and cascade results are benchmark-bound and may overstate gains for small calls.

### Cited Findings
- Anthropic offers efficiency-first (start at Haiku 5.5, upgrade if needed) and capability-first paths, says "most workloads start with Claude Opus 5.5", and lists subagent tasks under Haiku. A benchmark on your own data is "the most important step". Effort is "often a better lever than switching models" — [Anthropic choosing a model](https://platform.claude.com/docs/en/about-claude/models/choosing-a-model).
- A pairing must beat one model's whole score-versus-spend curve, not its default point. Recheck at each release. "Failed tasks still bill", so a cheap worker failing hard tasks can cost more. Opus 5.5 at medium with Haiku subagents cut cost 32% and lost about 4 points (95% CI 1.3–7.1), about what lower effort alone gives — [Anthropic, optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence).
- OpenAI recommends prototyping with the most capable model for every task to set a baseline and then swapping smaller models where targets hold — [OpenAI practical guide to building agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf) (2025). It frames cost per successful task as a better measure than token price — [OpenAI cost optimization guide, a voice-agent page seen via search excerpt](https://developers.openai.com/api/docs/guides/voice-latency-cost), and shows a smaller model for classification and extraction — [OpenAI cookbook](https://developers.openai.com/cookbook/examples/agent_optimization/optimizing_agents_for_cost_and_quality).
- Claude Code practice: safety classifier default is Sonnet 5, not Haiku and not the top model, and Opus 5, Sonnet 5.5 and Opus 5.5 are not valid overrides — [Claude Code permission modes](https://code.claude.com/docs/en/permission-modes). Prompt suggestions use the session's model for the cache ([costs](https://code.claude.com/docs/en/costs)). `ANTHROPIC_SMALL_FAST_MODEL` is marked deprecated ([prompt caching](https://code.claude.com/docs/en/prompt-caching)).
- Auto-mode cascade inside one judge: stage 1 FPR 8.5% and FNR 6.6%; full pipeline FPR 0.4% and FNR 17% on real overeager actions (n=52), on Sonnet 4.6 — [Anthropic engineering](https://www.anthropic.com/engineering/claude-code-auto-mode). This shows cascades trading recall for precision, and the figures may not reflect current models.
- FrugalGPT (May 2023, older): up to 98% lower cost matching GPT-4, or 4% better at equal cost — [arXiv 2305.05176](https://arxiv.org/abs/2305.05176).
- RouteLLM (June 2024, older): over 2x cost reduction in certain cases without quality loss, using preference data — [arXiv 2406.18665](https://arxiv.org/abs/2406.18665).
- Contrary evidence: kNN routers match or beat complex learned ones — [arXiv 2505.12601](https://arxiv.org/html/2505.12601v2). Seven of eight routers sit within a 0.85 point band and 27 points below oracle, suggesting routers mostly learn global averages — [Routing Plateau, arXiv 2606.07587](https://arxiv.org/html/2606.07587v1). Both preprints.
- A gateway-level confidence cascade exists, and Vercel advises setting thresholds from recorded errors. Unavailable confidence matches conservatively and triggers escalation — [Vercel decision fallbacks](https://vercel.com/docs/ai-gateway/models-and-providers/decision-fallbacks).

### Inferences
- Tier should be a property of the call, justified by the call's eval. Failover must not move a call to a lower tier if that tier failed the eval, the mirror image of Claude Code's rule against falling back to a smaller context window for compaction.
- "Cheap models first" for the router fits the bounded-task case, but the router's correct output includes a cost: a cheap wrong route may spend more downstream. Measure route accuracy and downstream cost together.
- Benchmark cascade gains do not transfer directly. They assume many queries where the cheap model suffices and a scorer exists. A daemon's 13 side calls are small and low volume, so the absolute savings may be small enough that reliability dominates.

### Gaps
- I could not retrieve OpenAI's current GPT-6 tier guide (403) or Google's tier guidance, so current OpenAI and all Google recommendations are incomplete.
- No published per-task evaluation of tiers for summarization, memory extraction or skill promotion.

## 6. Structured output reliability

### Takeaway
Native strict schema output is the best default where supported, but it guarantees shape only and has well-documented escape paths: refusals, truncation, enum casing, silently dropped keywords. Code must validate and treat those cases as first-class outcomes; older workarounds such as forced tool choice and prefill now return 400 on some models.

### Cited Findings
- Anthropic structured outputs use constrained decoding. Exceptions: `stop_reason: refusal` (HTTP 200, billed, may not match schema); `max_tokens` (may be incomplete); enum values may differ only by case, with no error. A property requesting step-by-step reasoning can trigger a `reasoning_extraction` refusal, so ask for a short explanation and leave reasoning in thinking blocks — [Anthropic structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).
- Anthropic: first request with a new schema pays compile latency, grammars cache for 24 hours from last use, and there is a 180-second compile timeout. Limits include 24 optional and 16 union-typed parameters; exceeding them returns 400 — [Anthropic structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).
- **Forced tool use (`tool_choice` any or tool) returns 400 on Opus 5.5, Sonnet 5.5 and Fable 5.1; assistant prefill returns 400 on Claude 4.6 and later.** Anthropic points to structured outputs or `auto` with strict tools — [Anthropic errors](https://platform.claude.com/docs/en/api/errors).
- Thinking cannot be turned off on Opus 5.5 or Fable models. Sonnet 5.5 offers `between_tools` as its lowest setting, and Haiku 5.5 accepts disabled — [Anthropic errors](https://platform.claude.com/docs/en/api/errors).
- OpenAI: JSON mode gives valid JSON only, while strict `json_schema` adheres to the schema. Check for a `refusal` and for `incomplete` due to `max_output_tokens` before parsing. Key order follows the schema, and the docs' reasoning pattern is a `steps` array followed by `final_answer` — [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
- Gemini: syntactic validity and schema structure are guaranteed, semantics are not. `minLength`, `maxLength`, `uniqueItems` and others are **silently stripped**, while `minItems` and `maxItems` are enforced — [Gemini structured outputs](https://aistudio.google.com/docs/structured-output).
- Tam et al. (EMNLP 2024, older models) found format restrictions degraded reasoning while often helping classification — [paper](https://aclanthology.org/2024.emnlp-industry.91). dottxt published a rebuttal disputing the methodology ("Say What You Mean"), which I did not read in full; search excerpts only ([reference](https://frontpage-app.vercel.app/news/46345333)).
- Malformed answers can still be billed, and Vercel returns 500 on a malformed primary decision while billing the stage — [Vercel decision fallbacks](https://vercel.com/docs/ai-gateway/models-and-providers/decision-fallbacks). Hermes counts malformed or empty 200 responses as failures that can trigger fallback — [Hermes](https://hermes-agent.nousresearch.com/docs/user-guide/features/fallback-providers).

### Inferences
- Outcome type for a helper should be something like: `Valid<T>`, `Refused`, `Truncated`, `SchemaInvalid`, `ProviderError`. The first failure should not be flattened into "parse failed".
- Providers conflict on where reasoning goes. A schema-first reasoning field is reasonable on OpenAI-style strict output and risks a refusal on Anthropic, so the schema must be per provider or leave reasoning out of the output. That is a further hidden cost of cross-provider failover for structured calls.
- Always-on thinking means tiny output caps can truncate valid work; test each call's `max_tokens` on each model in its candidate list.
- Compile latency makes a short shared deadline a trap on the first call with a new schema, so a warm-up or generous first-call handling may be needed.
- Same-schema across providers is not same semantics: a keyword may be enforced by one and stripped by another. Local validation of constraints should be the rule.

### Gaps
- No current independent numbers comparing repair, re-ask and strict mode on the current generation of models.
- Handling of reasoning fields under Gemini structured output was not documented on the page read.

## 7. Telemetry, silent non-execution, evals and shadow testing

### Takeaway
OTel GenAI gives a usable (still Development-status) attribute set but no purpose attribute, so purpose needs a local convention. Silent non-execution is only detectable by comparing expected firings to observed outcomes, including skips and last-success timestamps, and no LLM-specific standard for that exists.

### Cited Findings
- The GenAI conventions moved to a dedicated repository, and the old page is unmaintained — [OTel GenAI spans (moved notice)](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/). They are marked Development — [semconv repo](https://raw.githubusercontent.com/open-telemetry/semantic-conventions-genai/main/docs/gen-ai/gen-ai-spans.md).
- Inference spans: required `gen_ai.operation.name` and `gen_ai.provider.name`; conditionally `error.type`, `gen_ai.request.model`, `gen_ai.output.type`, and **`gen_ai.prompt.name` and `gen_ai.prompt.version` when a named prompt is used**; recommended usage with `cache_read`, `cache_write` and `reasoning` token subsets, `finish_reasons`, `response.model`. A missing expected finish reason should be reported as `error` — [OTel client inference](https://raw.githubusercontent.com/open-telemetry/semantic-conventions-genai/main/docs/gen-ai/client-inference.md).
- On retries: the `fetch_response` span definition says that for an automatically retried transient failure the span should cover the whole logical operation including retries, and no retry attribute is defined; I did not confirm identical wording for inference spans, though their time-to-first-chunk measure explicitly includes failed attempts and backoff — [OTel spans](https://raw.githubusercontent.com/open-telemetry/semantic-conventions-genai/main/docs/gen-ai/gen-ai-spans.md).
- Metrics include `gen_ai.client.operation.duration` with `error.type` on failure. `invoke_agent.inference_calls` counts failed calls and counts a logical operation once regardless of retries — [OTel metrics](https://raw.githubusercontent.com/open-telemetry/semantic-conventions-genai/main/docs/gen-ai/gen-ai-metrics.md).
- Claude Code as a worked example: `query_source` purpose tag, an `attempt` count on the span plus a `gen_ai.request.attempt` event per retry, `error_class`, `api_refusal` as a distinct event because no HTTP error occurs, and cost and tokens by `query_source` with cache types split. It has no API error counter metric — [Claude Code monitoring](https://code.claude.com/docs/en/monitoring-usage).
- Vercel records every model and provider attempt with status, and per-attempt usage and cost on decision fallbacks — [Vercel docs](https://vercel.com/docs/ai-gateway/models-and-providers/model-fallbacks).
- Prometheus: export a default 0 for series known in advance; track failures alongside totals; for batch jobs the key metric is the last success time as a gauge — [Prometheus instrumentation](https://prometheus.io/docs/practices/instrumentation/).
- SRE: test the degraded path regularly, and alert on entry to degraded mode — [SRE, cascading failures](https://sre.google/sre-book/addressing-cascading-failures/).
- Anthropic evals: multiple trials per task; pass@k and pass^k; capability versus regression suites; 20–50 tasks drawn from real failures is a good start; run in CI; layer automated evals with production monitoring and transcript reading — [Anthropic evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).
- Shadow: LiteLLM's `silent_model` mirrors requests in the background without affecting the primary — [LiteLLM traffic mirroring](https://docs.litellm.ai/docs/traffic_mirroring). SageMaker shows the general pattern with a sampling percentage — [AWS docs](https://docs.aws.amazon.com/sagemaker/latest/dg/shadow-tests.html).

### Inferences
- `gen_ai.prompt.name` is the closest standard slot for "which of the 13 calls". Add a local attribute such as `aidaemon.call.purpose` with `aidaemon.call.policy`, since the standard has no purpose concept and the conventions may still change.
- One logical span per call plus an attempt event per try, as Claude Code does, satisfies OTel and retains failover detail.
- Record every outcome, including `skipped_no_budget`, `gave_up_quietly`, `cancelled_by_deadline`, `disabled_by_config`, `refused`, and whether a call failed before billing. Then compute per-purpose expected versus observed: for every eligible trigger there must be exactly one terminal record.
- Detection options for a call that never runs: pre-registered zero-valued counters per purpose; last-success timestamp per purpose with an alert appropriate to its cadence; a CI test that exercises each call site through the shared path and asserts one record. aidaemon already has `db_probe --gate-outcomes` as a model of outcome-labeled measurement.
- Shadowing side-effect-free one-shot calls is unusually safe, but duplicates content to the candidate model (a data-handling question) and doubles cost. For the helper migration, compare old and new paths on recorded inputs first and sample live traffic only for non-sensitive classes.

### Gaps
- No LLM-specific standard or published recipe for missing-execution detection was found; the recipe above is adapted from general monitoring practice.
- Whether OTel expects token usage on failed inference spans was not stated on the portions read, and the token-metrics document was not read.

## 8. Criteria for keeping or deleting an auxiliary call

### Takeaway
Anthropic's published approach is one-at-a-time ablation against final outcomes, re-run as models change. Pass-rate differences need repeated runs and sample sizes sized to the effect, and safety calls should be judged by replay over labeled cases rather than live removal.

### Cited Findings
- Anthropic's harness work: a radical cut lost quality and hid which parts were load-bearing, so the author removed one component at a time. The sprint construct was removed; the planner was kept; the evaluator was reduced to one end-of-run pass and is "worth the cost when the task sits beyond what the current model does reliably solo". Each component encodes an assumption about what the model cannot do, and such assumptions "can quickly go stale" — [Anthropic harness design](https://www.anthropic.com/research/harness-design-long-running-apps).
- Compare any pairing against the same model at lower effort and against the next model down. Recheck at each release — [Anthropic, optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence).
- That same page shows an advisor that was consulted in 0 of 198 questions on one benchmark and 1 of 300 tasks on another, scoring within noise or below the executor alone, while still adding token overhead — [same page](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence). This is a direct example of a component that silently does nothing.
- A/B testing measures real outcomes but is slow and needs traffic; production monitoring is reactive; small samples can detect large early effects — [Anthropic evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).
- An academic ablation study holds supporting components fixed across conditions to isolate one intervention — [arXiv 2609.20804](https://arxiv.org/pdf/2609.20804) (preprint; seen via search excerpt only).
- Auto mode is evaluated by false positive and false negative rates on separate datasets (n=10,000 real traffic, n=52 real overeager actions, n=1,000 synthetic exfiltration) — [Anthropic engineering](https://www.anthropic.com/engineering/claude-code-auto-mode).

### Inferences
- Suggested decision record for each of the 13 calls: outcome it protects; does it run, and how often; does its output change behavior; ablation result; replacement by deterministic logic or typed state (aidaemon's "evidence beats contract" rule favors this); cost per useful decision; failure consequence.
- Deletion is favored when a call rarely fires or rarely changes the result, when a deterministic or receipt-based signal gives the same answer, or when N-of-M runs without it are indistinguishable at a defined margin. Keep when removal changes a measurable outcome, or when the call is a safety boundary whose loss cannot be observed by outcome metrics.
- Safety judges should not be A/B tested by removal in live traffic. Replay against a labeled corpus and report FPR and FNR per candidate model. Mind sample size: auto mode's 17% FNR on n=52 has an approximate 95% Wilson interval of 9–30% by my calculation, assuming 9 of 52.
- N-of-M sizing, standard statistics not from a cited source: 0 failures in N only bounds the failure rate below about 3/N at 95% for large N; the exact one-sided bound for 8/8 is a true failure rate up to roughly 31%. Distinguishing 90% from 70% at 80% power takes roughly 60 runs per arm. An 8-run comparison detects only large effects such as 8/8 versus 1/8.
- Ablate through the shared path, with a skip outcome recorded, so that a removed call is visible in telemetry and can be restored. A deletion shipped behind an off switch matches SRE's advice to keep a quick way to disable or retune complex behavior ([SRE](https://sre.google/sre-book/addressing-cascading-failures/)).

### Gaps
- No published kill criteria or thresholds specifically for auxiliary LLM calls. The record above is a synthesis.
- No study on how often auxiliary calls in shipped agents turn out to be removable.
