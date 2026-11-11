# Model calls outside the main agent loop: routing, compaction, background summaries, diagnostics

Research date: 2026-11-09. "Project note" means the repository owner's local record, not a public source. Items marked [search only] were seen in a search result or aggregator, and the full source was not read. Paper results are from preprints unless stated otherwise.

## Do leading agents use a pre-turn intent or continuation classifier, or leave that to the main model with full history?

### Takeaway
I found no public evidence that Claude Code or Codex classifies each turn's relationship to earlier turns. The documented coding-agent default is to send the history and let the main model interpret it. Pre-turn classifiers are well documented in customer-service routing and tool selection, but no source tests a continuation classifier against a full-history main model. The aidaemon A/B also does not test that comparison.

### Cited Findings

**Vendor guidance**
- Anthropic says routing fits complex tasks with distinct categories that are better handled separately, and gives model-tier routing as an example. It also says to add complexity only when it demonstrably improves outcomes. The post dates from 2024-12-19, and the page says much of the tooling has changed — [Anthropic, Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- Its router is a dispatch pattern that selects a follow-up path. It does not describe a classifier that decides what history a single general agent receives — [Anthropic, Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- OpenAI says simple retrieval or intent classification may suit a smaller, faster model. It recommends prototyping every task with the most capable model to set a baseline, then swapping in smaller models and checking results — [OpenAI, A practical guide to building agents (PDF)](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)
- OpenAI recommends getting the most from a single agent before adding agents, because more agents add complexity and overhead. Its guardrail list includes a relevance classifier for keeping responses in scope — [OpenAI, A practical guide to building agents (PDF)](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)

**How production agents behave**
- Claude Code sends the system prompt, project context, every prior message, every tool result and the new message on each request, and depends on prefix caching to make that affordable — [Claude Code, prompt caching](https://code.claude.com/docs/en/prompt-caching)
- Claude Code documents side calls for titles, recaps, prompt suggestions and compaction. The pages I read list no per-turn relationship classifier. Session titles use the small/fast model — [Claude Code, costs](https://code.claude.com/docs/en/costs); [Claude Code, env vars](https://code.claude.com/docs/en/env-vars)
- Claude Code does run a separate classifier before actions in auto mode, on Sonnet 4.6, with assistant prose and tool outputs removed from its input. Stage 1 is a fast filter, and stage 2 reasons only about flagged actions and mostly reuses stage 1's cache. On real traffic (n=10,000), false positives fell from 8.5% at stage 1 to 0.4% for the full pipeline. On real overeager actions (n=52), the full pipeline missed 17% — [Anthropic, Claude Code auto mode (2026-03-25)](https://www.anthropic.com/engineering/claude-code-auto-mode)
- LangChain ships an optional pre-model LLM tool selector for agents with many tools (its docs suggest 10 or more). It defaults to the main model, and the docs show a smaller model as the option — [LangChain, built-in middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in)

**Measured classifier results**
- In an Amazon multi-turn e-commerce system, fine-tuned RoBERTa beat zero-shot LLMs on accuracy. P50 latency was 0.08 s for RoBERTa, 1.98 s for Nova Pro zero-shot, 5.93 s for Claude Sonnet 3.7 zero-shot, and 1.99 s for a hybrid that used the LLM only on low-confidence cases. The hybrid had the best accuracy because the classifier narrowed the LLM's choices. The paper reports a 7.89% gain in positive customer responses and production P50 of 0.12 s — [Xie et al., 2026](https://aclanthology.org/2026.customnlp4u-1.8.pdf)
- That paper names intent switching within conversations as a core difficulty, and had to simulate dialogues because topic switches were rare in real transcripts — [Xie et al., 2026](https://aclanthology.org/2026.customnlp4u-1.8.pdf)
- MARCO classifies each query as Info, Action or out-of-domain, then passes the conversation history to a multi-agent task layer. The full system reached 94.48% and 92.74% accuracy at 5.61 s and 5.85 s with guardrails, and reports a 33.71% cost reduction against a single-agent baseline. These are system numbers from 2024-era models, not classifier-only numbers — [MARCO, Amazon](https://arxiv.org/pdf/2410.21784)
- A third-party repository reports fine-tuned XLM-RoBERTa beating Claude Sonnet 4.6 on BANKING77 (94.2% vs 85.3%, about 12 ms vs 2 s). This is single-turn and not peer reviewed [search only] — [intent-router](https://github.com/smallestbusiness/intent-router)

**Costs of each side**
- Each model has its own cache. Claude Code's team says that 100k tokens into an Opus session, sending an easy question to Haiku costs more than letting Opus answer, because Haiku must rebuild the cache. Cached input is about one-tenth the uncached price — [Claude Code team, Prompt caching is everything (2026-04-30)](https://claude.dev/blog/lessons-from-building-claude-code-prompt-caching-is-everything/)
- Full history has its own failure mode. Across six generation tasks and more than 200,000 simulated conversations, LLMs did 39% worse in multi-turn than single-turn settings, and early wrong assumptions were hard to recover from. The study dates from May 2025 and uses underspecified, shard-by-shard instructions — [Laban et al.](https://arxiv.org/abs/2505.06120)

**The local evidence**
- With 8 runs per scenario per arm, the router-on arm scored 8/8 vs 0/8 on tool-only detail follow-ups, 8/8 vs 5/8 on retry, 8/8 vs 6/8 on resume, and 8/8 vs 0/8 on closing an unfinished request. Unrelated new requests were 8/8 in both arms — Project note: relationship-router-ab-2026-11-08
- The note attributes the 0/8 to the off arm having no tool history, which made the agent falsely deny having run the tool. After same-day fixes, including a per-turn note of which recent turns ran tools and opening tool output by turn id, every scenario passed 8/8, including follow-up after two unrelated turns, which had been 0/8 in both arms — Project note: relationship-router-ab-2026-11-08

### Inferences
- The A/B shows that aidaemon needs a relationship signal, or something that provides its effect, given how a turn is assembled when no relationship applies. It does not show that a classifier beats full history. The missing arm is router off, with the tool-history note and `search_history` open-by-turn always available.
- Pressure on the design is mostly about failure direction. A false `new_request` can cause a false denial, which is costly. A false `continuation` adds some context. The router's error rate should be measured per direction, as Anthropic does for auto mode, and not as one accuracy figure.
- "Failure means no relationship is applied" is the riskier default, because it produces the same state as the off arm. Keep the failure mode, but have the no-relationship path still expose recent tool activity. Worth testing by fault injection.
- The 45 second shared deadline is much longer than the 2 to 6 second LLM classifier latencies published. Treat it as a worst case and report p50, p95 and the rate of hitting the deadline.
- A cheap router should send a short, bounded candidate list and never the whole history, because a different model gets no cache benefit from the main conversation.
- Candidate text from earlier turns can be untrusted. Auto mode strips tool output from its classifier for this reason. Check whether any router input includes tool results or other people's content.
- A hybrid is plausible but premature. Typed signals such as an open request, a recent tool run or an explicit reply-to could resolve easy cases without a model call, in line with the project's structural-signal rule. The Amazon result supports narrowing the model's choices, but its setting is customer-service intent.

### Gaps
- No public A/B compares a continuation classifier with full-history main-model handling. No production evidence for contextual query rewriting was found.
- Absence of a documented classifier in Claude Code does not prove there is none. Codex's turn logic was not audited for this question.
- The publication date of the OpenAI guide was not visible in the extracted text.

## Compaction: what do Claude Code, Codex, Google ADK, LangChain/LangGraph and Anthropic's APIs do, and what is lost?

### Takeaway
Production systems mostly compact at a threshold, on the session's own model where the prefix cache can be shared, and then restore critical state deterministically. Background compaction is now a first-party API pattern. Recent benchmarks show LLM summaries keep few literal details, that summarizer choice can change downstream failure rates substantially, and that task completion alone hides the cost.

### Cited Findings

**Product behavior**

| System | Trigger and timing | Summarizer | Structure and what is kept |
|---|---|---|---|
| Claude Code | Auto near the limit, manual `/compact`, and idle compaction on one-hour-TTL sessions | Separate request repeating the system prompt, tools and history; thinking follows session settings | Summary plus re-injected instructions, memory, plan, up to 5 recent files and capped skill bodies |
| Codex CLI (main, read 2026-11-09) | Inline auto task with phases including post-turn; remote or local by provider capability | Turn's context for local compaction, with a configurable prompt | Handoff summary plus retained user messages up to 20,000 tokens |
| Anthropic API | On demand, in the background, or at a token threshold | Claude writes it on the server | `<summary>` block; optional kept recent turns |
| Google ADK | Token threshold or sliding interval, run by the Runner in the background | Optional; docs example `gemini-flash-latest` | Recent raw events and overlap events remain |
| LangChain | Middleware trigger on fraction, tokens or messages; no automatic trigger unless set | Any model; docs example is a smaller model | Keeps 20 recent messages by default |

Sources for the table: [Claude Code, context window](https://code.claude.com/docs/en/context-window); [Claude Code, prompt caching](https://code.claude.com/docs/en/prompt-caching); [Codex compact.rs](https://github.com/openai/codex/blob/main/codex-rs/core/src/compact.rs); [Codex compact task](https://github.com/openai/codex/blob/main/codex-rs/core/src/tasks/compact.rs); [Anthropic, compaction overview](https://platform.claude.com/docs/en/build-with-claude/compaction); [Google ADK, compaction](https://adk.dev/context/compaction/); [LangChain, built-in middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in)

**Details**
- Claude Code's cache-safe compaction reuses the parent's system prompt, tools and messages and appends the summarize instruction, so the prefix stays cached. A tool-less custom prompt would miss the cache entirely. The summary is about 20k tokens, and the design reserves a buffer for it — [Claude Code team, Prompt caching is everything](https://claude.dev/blog/lessons-from-building-claude-code-prompt-caching-is-everything/)
- After a break longer than the cache lifetime, the summarization request is uncached, so compaction costs most on resumed sessions. Claude Code advises compacting at natural breaks — [Claude Code, prompt caching](https://code.claude.com/docs/en/prompt-caching)
- After compaction, Claude Code re-reads up to five files. Files over 5,000 tokens return as path references. Skill bodies are capped at 5,000 tokens each and 25,000 in total. Running background tasks are listed so they are not duplicated. Path-scoped rules are summarized away until reloaded — [Claude Code, context window](https://code.claude.com/docs/en/context-window)
- Anthropic recommends on-demand compaction wherever available. Beta options include keeping recent turns verbatim and running in the background — [Anthropic, compaction overview](https://platform.claude.com/docs/en/build-with-claude/compaction)
- For background compaction, the application records how many messages it sent, continues on full history, allows only one compaction at a time, then replaces exactly those messages when the block arrives. Start early enough that the window can hold the turns that arrive meanwhile — [Anthropic, background compaction](https://platform.claude.com/docs/en/build-with-claude/compaction-background)
- Threshold compaction defaults to 150,000 input tokens, minimum 50,000. Custom instructions replace the default prompt. Compaction iterations are billed but excluded from top-level usage. If no summary is produced, keep the original messages — [Anthropic, threshold compaction](https://platform.claude.com/docs/en/build-with-claude/compaction-threshold)
- Tool-result clearing defaults to a 100,000-token trigger and keeps 3 tool uses. It invalidates the cached prefix, so `clear_at_least` exists to make each clear worth the rewrite. The docs report no benchmark — [Anthropic, context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing)
- The deprecated client-side compaction allowed a cheaper model such as Haiku and used sections for Task Overview, Current State, Important Discoveries, Next Steps and Context to Preserve — [Anthropic, context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing)
- Codex's default prompt asks for a handoff to another LLM covering progress, decisions, constraints, remaining steps and critical references. When compaction itself overflows the window, Codex removes the oldest history item and retries, with a code comment saying this preserves the prefix cache — [Codex prompt](https://github.com/openai/codex/blob/main/codex-rs/prompts/templates/compact/prompt.md); [Codex compact.rs](https://github.com/openai/codex/blob/main/codex-rs/core/src/compact.rs)
- Third-party write-ups disagree on Codex's default auto-compact token limit (about 200,000 vs 64,000), so neither should be relied on [search only] — [Vaughan, Codex compaction tuning](https://codex.danielvaughan.com/2026/04/16/codex-cli-context-compaction-tuning-long-sessions/)
- Anthropic's tuning advice is to maximize recall first, then remove superfluous content. It calls tool-result clearing one of the safest forms of compaction (2025-09-29) — [Anthropic, Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- OpenAI's cookbook contrasts deterministic trimming (no added call or summary drift, but older constraints vanish) with summarization (long-range recall, but added latency and possible context poisoning). Its example checks contradictions, marks uncertain facts, uses fixed headings, releases a lock while summarizing and rechecks before applying. It is undated, and defaulting to `gpt-4o` suggests it is older — [OpenAI cookbook, session memory](https://developers.openai.com/cookbook/examples/agents_sdk/session_memory)
- Manus argues that compression should be restorable, such as dropping page content while keeping its URL, because irreversible compression is risky when later relevance is unknown. It treats cache hit rate as the key production metric (2025-07-18) — [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- Cognition proposes a dedicated compression model, says it is hard to get right, and reports fine-tuning a smaller model for it (2025-06-12) — [Cognition](https://cognition.com/blog/dont-build-multi-agents)

**Evaluation and measured loss**
- Compact-Bench pairs full-history continuation with summary-only continuation from the same repository state, over 35 SWE-bench repairs, 146 sessions and 106 production compaction chains. Among 22 full-history successes, summary-only continuation failed 5% (Claude Opus 4.8) to 64% (DeepSeek-Flash) of the time — [Compact-Bench (Oct 2026)](https://arxiv.org/html/2610.31710)
- LLM summaries retained at most 16.0% of recurring details, against 79.1% for a mechanical anchor index. Production chains averaged 15.7%, 19.9%, 4.9% and 1.8% at the first through fourth boundaries, though the later samples are small — [Compact-Bench](https://arxiv.org/html/2610.31710)
- Retention and success diverge. DeepSeek-Flash had top retention with the Claude Code prompt but the worst damage. In failed repairs, summaries kept target files and code strings but lost the reasoning needed to finish the edit — [Compact-Bench](https://arxiv.org/html/2610.31710)
- A controlled study found compression increased retrieval tool calls in all six model-regime comparisons while completion changed non-significantly at 5x. GPT-5.5 went from 80% to 85% completion while retrieval rose by 42.9 calls. A fact-preserving operator at the same ratio avoided most of the extra retrieval. A second environment (ALFWorld) showed no surge — [Liu, What Does Context Compression Cost an Agent? (Aug 2026)](https://arxiv.org/pdf/2608.16370)
- Factory's probe method ran recall, artifact, continuation and decision probes over 36,611 production messages. Scores out of 5 were Factory 3.70, Anthropic 3.44 and OpenAI 3.35. All methods were weakest on artifact trail (2.19 to 2.45). This is a vendor evaluation of its own product, judged by an LLM (2025-12-16) — [Factory](https://factory.com/news/evaluating-compression)

### Inferences
- Aidaemon's after-reply background pass matches current first-party practice. Its race condition should be tested with the Anthropic invariant: replace exactly the messages summarized, never drop turns added during the call, and allow one job at a time.
- The fast-model choice is the least supported part of the summarizer. Claude Code, the Anthropic API and Codex's local path effectively use the working model or its context, while ADK and LangChain examples use cheaper models. Compact-Bench shows choice of backbone can matter a great deal, without showing that cheap models are always worse. Run a replay comparing fast-model and primary-model summaries.
- A separate fast-model call cannot read the main conversation's cache. Cost should be measured against a cached same-model fork. The answer may differ for aidaemon's ChatGPT-subscription primary path, where latency and quota matter more than list price.
- Source handles are the design's best-supported feature. They match restorable compression and address the weakest dimension in Factory and Compact-Bench. Test that handles resolve and that the agent actually opens them after compaction.
- Add a deterministic state block beside the prose: paths, IDs, commands, errors, open obligations and receipts. Mechanical retention is far better than prose, and Compact-Bench suggests rationale must also be retained, so neither replaces the other.
- Measure re-retrieval calls, repeated tool calls and false "never ran it" answers, because completion can stay flat. Also check `db_probe --cache-reuse` around each summary swap, since each rewrite can force a rebuild.
- The 30 second emergency pass needs a defined fallback on timeout. Codex's oldest-first trimming is a useful model: deterministic, and it keeps recent context.
- Repeated compaction should be a named test, since retention falls sharply across boundaries in the small late samples.

### Gaps
- Docs do not explicitly name the model that writes Claude Code or API summaries. Same-model use is inferred from the cache-sharing design and API examples.
- I found no first-party or research treatment of taint flags that persist through summaries.
- No study isolates after-reply background compaction against at-threshold compaction on quality or user-perceived latency.
- LangGraph-specific behavior beyond LangChain middleware was not checked.

## When is a tool-internal LLM call good or bad design, and should tools return raw evidence or pre-digested judgment?

### Takeaway
Tool-internal model calls are accepted when they reduce large material to something the caller can use, when their lossiness is documented, and when raw access remains available. They are weaker when the tool's input is already small, the callee is cheaper than the caller, and the output is a judgment the caller could make itself. No controlled study was found comparing an internal-LLM tool to a raw-evidence tool.

### Cited Findings

**Supporting internal model calls**
- Claude Code's WebFetch usually runs the requested extraction in a separate model call, and the agent receives that result, not the page. Its docs call this lossy by design and point to `curl` for the unprocessed page — [Claude Code, tools reference](https://code.claude.com/docs/en/tools-reference)
- Anthropic's advisor tool lets a cheaper executor consult a stronger model that reads the full transcript, runs without tools and returns advice. Errors do not fail the request. Anthropic reports task-dependent results, including a gain of about 7 pass-rate points for a Haiku nudge, and no measurable effect or a slight loss for stronger executors — [Anthropic, advisor tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool)
- Sub-agents may spend tens of thousands of tokens and return 1,000 to 2,000 tokens to the lead agent — [Anthropic, Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- Claude Code recommends delegating verbose operations such as test runs and log processing to subagents so that only a summary returns — [Claude Code, costs](https://code.claude.com/docs/en/costs)

**Against, or favoring raw and referenced evidence**
- Cognition argues for sharing full agent traces rather than summaries or individual messages, because nuance is otherwise lost (2025-06-12) — [Cognition](https://cognition.com/blog/dont-build-multi-agents)
- Anthropic's research system has subagents write to external stores and return lightweight references, to avoid a game of telephone through the lead agent. It also reports about 15x the tokens of chat for multi-agent systems — [Anthropic, multi-agent research](https://www.anthropic.com/engineering/multi-agent-research-system)
- Programmatic tool calling reduced average tokens by 37% (43,588 to 27,297), with internal retrieval accuracy rising from 25.6% to 28.5% and GIA from 46.5% to 51.2%. Anthropic says it helps less when the model must see all intermediate results or for small lookups (2025-11-24) — [Anthropic, advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use)

**Tool response design**
- Tools should return high-signal information, can expose `response_format` for concise or detailed output, and should use filtering, pagination and truncation. Claude Code limits tool responses to 25,000 tokens by default. In one example, concise output was 72 tokens and detailed output 206 — [Anthropic, Writing tools for agents (2025-09-11)](https://www.anthropic.com/engineering/writing-tools-for-agents)
- That article endorses tools that consolidate work and return relevant log lines or compiled context. It does not discuss tools that call a model internally — [Anthropic, Writing tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Natural-language names or simple indexes improved retrieval precision over opaque IDs in Anthropic's evaluation — [Anthropic, Writing tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Deterministic preprocessing can reduce a large log read from tens of thousands of tokens to hundreds, as in the grep example for hooks — [Claude Code, costs](https://code.claude.com/docs/en/costs)
- Rewriting tool descriptions from a tool-testing agent's diagnoses cut future task completion time by 40% — [Anthropic, multi-agent research](https://www.anthropic.com/engineering/multi-agent-research-system)

### Inferences
- A useful test for an internal LLM is: does it remove far more context from the caller than it adds, and can it do something the caller cannot do cheaply? WebFetch passes on page size. Advisor passes by applying a stronger model. Diagnose ranking uses a cheaper model to reconsider at most 12,000 characters, roughly 3,000 tokens at a rough four characters per token, for a caller that may be stronger.
- Reports should carry evidence IDs and clearly separate deterministic findings from model opinion, so the caller can check them.
- The Anthropic and Cognition positions differ less than they first appear. Both reject unverifiable summaries, and they differ on when isolation is worth the loss. Reference-returning summaries fit both.

### Gaps
- I found no controlled comparison of the same tool returning raw evidence, deterministic digests, and an LLM digest. Reports on summary loss in deep-research agents were secondary and not usable here.

## How should user-facing notifications or summaries of background work be produced?

### Takeaway
Public evidence favors structured events and fixed status text for machine-known facts, usually re-entering the agent when interpretation is needed. Short model-written session recaps exist, but I found no leading agent documented as using a dedicated one-sentence, tool-less completion interpreter, and no study comparing templates with model-written notices.

### Cited Findings
- Claude Code's background Bash returns a task ID, writes output to a file and lets the agent read it. It also uses fixed-format stop notices for time-limit stops — [Claude Code, interactive mode](https://code.claude.com/docs/en/interactive-mode); [Claude Code, tools reference](https://code.claude.com/docs/en/tools-reference)
- Claude Code's Monitor tool feeds each output line back to Claude so it can react during the conversation — [Claude Code, tools reference](https://code.claude.com/docs/en/tools-reference)
- After compaction, Claude Code tells the model which background commands are still running so it does not start duplicates. This is a direct, deterministic guard against re-execution — [Claude Code, context window](https://code.claude.com/docs/en/context-window)
- Third-party SDK documentation describes a `task_notification` carrying terminal status, an output summary and usage statistics [third-party] — [ClaudeCode Elixir SDK](https://hexdocs.pm/claude_code/ClaudeCode.Message.SystemMessage.TaskNotification.html)
- Issue reports describe regressions where background completion failed to notify or wake the agent, so delivery reliability is a separate failure mode. Those reports are marked resolved [search only] — [issue mirror](https://claudeissues.com/issue/20688-bug-regression-background-commands-dont-notify-claude-code-on-completion)
- Claude Code's session recap is a one-line background summary shown after at least three minutes away, needs at least three turns, is capped at 400 characters and never appears twice in a row. `/recap` appends output and leaves the cached prefix intact — [Claude Code, interactive mode](https://code.claude.com/docs/en/interactive-mode); [Claude Code, prompt caching](https://code.claude.com/docs/en/prompt-caching)
- Background functionality in Claude Code can use a Haiku-class model, and session titles do. The pages read do not name the recap model — [Claude Code, env vars](https://code.claude.com/docs/en/env-vars)
- In a third-party description, Codex notifications pass the last assistant message to a notify hook, so the user-visible text comes from the agent's turn. A commit in a fork shows a structured end event with an exit code for background processes [third-party] — [Vaughan, Codex notification pipeline](https://codex.danielvaughan.com/2026/04/13/codex-cli-notification-pipeline-osc9-hooks-alerts/); [fork commit](https://forge.lthn.ai/core/core-agent-ide/commit/b5dd1890671d58dd92c29dc4ffa6432ecb48d54c)
- The project has moved to hosted models only and treats small-model scaffolding as drop-not-port — Project note: no-local-models-2026-11

### Inferences
- The interpreter's stated reason, that small models re-ran the command when re-entering the loop, may no longer apply under the hosted-only policy. Re-test re-entry on current models before keeping it for that reason.
- Exit status, command and duration should come from code. For short output, a template such as "`cmd` finished with exit 0: <output>" cannot misstate the result. Compare it with the current sentence in a small user-facing replay, scoring wrong success or failure claims and whether users had to ask a follow-up.
- If a model sentence stays, make the exit code and failure state deterministic inputs, ensure the sentence cannot contradict them, and retain raw output by reference so a later "what did it say?" finds it. The router A/B's false-denial failure suggests the notice must leave a discoverable record of what ran.
- Shell output can contain instructions. A tool-less call limits what injected text can trigger, but it can still mislead the user. Add an adversarial-output test.
- A short run with no owning task and no needed follow-up is the case that least justifies either a full agent turn or a model sentence.

### Gaps
- I found no measured comparison of template and LLM-written background notifications, and no public statement of which model writes Claude Code recaps.
- Codex's current user-facing completion path was not confirmed in first-party source.

## Does LLM re-ranking on top of deterministic root-cause candidates add value?

### Takeaway
There is real supporting evidence, but in telemetry RCA and with richer per-candidate evidence than aidaemon's call provides. The best recent paper shows re-ranking can beat fixed rules, that retrieval sets a hard ceiling, and that domain knowledge matters. I found nothing on agent-trace failure diagnosis or on 60/40 confidence blending.

### Cited Findings
- A September 2026 preprint separates retrieval failure (the cause is never considered) from reranking failure. Its single-call LLM reranker over structured per-candidate evidence, with one configuration, matched or exceeded the best baseline's top@1 on all six benchmark suites, by up to 12 points — [Muhammad et al., Where RCA Fails](https://arxiv.org/pdf/2609.36686)
- Selected top@1 results for the best baseline (BARO) vs the LLM without domain knowledge were WADI 0.214 to 0.333, SWaT 0.194 to 0.213, HVAC 0.000 to 0.153, RE1-OB 0.784 to 0.875, RE1-SS 0.856 to 0.872, and RE1-TT 0.560 to 0.653. The reranker was gpt-oss-120b, with K=15 and n=3 runs — [Muhammad et al.](https://arxiv.org/pdf/2609.36686)
- On the same candidates, no fixed rule generalized across benchmarks. The LLM beat or tied every rule and fusion on five of six, and on the sixth once domain knowledge was added — [Muhammad et al.](https://arxiv.org/pdf/2609.36686)
- With the true cause guaranteed in the pool, adding a short system-description document let the reranker lead the best baseline by 7 to 18 points — [Muhammad et al.](https://arxiv.org/pdf/2609.36686)
- Retrieval is a hard ceiling: a reranker cannot recover a candidate it was not given. On industrial benchmarks, 33 to 38% of causes never reached the reranker — [Muhammad et al.](https://arxiv.org/pdf/2609.36686)
- The authors warn about nondeterminism, automation bias and miscalibrated reranker confidence. They recommend a short ranked list with per-candidate evidence, not a single answer. Benchmarks with propagating faults are small — [Muhammad et al.](https://arxiv.org/pdf/2609.36686)
- OpenRCA covers 335 failures and more than 68 GB of telemetry. The best model solved 11.34% of cases even with a purpose-built agent (ICLR 2025, older models, raw-data task). OpenRCA 2.0 reportedly gets 20.7% exact root-cause sets, 76.0% at-least-one service, and 61.5% causal path reachability [search only] — [OpenRCA](https://mlanthology.org/iclr/2025/xu2025iclr-openrca); [OpenRCA 2.0](https://tldr.takara.ai/p/2606.27154)
- RCAEval supplies 735 failures with reproducible baselines, and a newer benchmark scores localization largely deterministically to avoid LLM-as-judge [partly search only] — [RCAEval](https://arxiv.org/pdf/2412.17015); [Multi-dataset benchmark](https://arxiv.org/html/2606.29193v1)
- Verbalized confidence is often overconfident, and a 2026 study reports that it barely separates plausible wrong answers from correct ones. ConfidenceBench reports best Brier scores of 0.103 among frontier models, with results varying by model [search only] — [Xiong et al.](https://arxiv.org/pdf/2306.13063v1); [arXiv 2605.27752](https://arxiv.org/html/2605.27752); [ConfidenceBench](https://arxiv.org/pdf/2607.20526)
- In aidaemon's `llm_rank_candidates`, categories are limited to deterministic candidates, the call uses the cheap model, parse failure falls back to deterministic output, and model text replaces a candidate's description when non-empty. Only candidates the model returns get the 60/40 blend — [src/tools/diagnose.rs](../../src/tools/diagnose.rs)
- The same function accepts `resp.thinking` when content is missing, so reasoning text could be parsed as the answer — [src/tools/diagnose.rs](../../src/tools/diagnose.rs)

### Inferences
- The evidence justifies evaluating the model step, not assuming it. Aidaemon's input differs from the winning setup: a truncated timeline and a baseline report, with no explicit system-description document and no per-candidate evidence packet.
- Because categories come only from deterministic candidates, retrieval failure cannot be repaired. Measure whether the true category is in the set before tuning rank.
- The 60/40 blend has no empirical support in the sources found. It mixes an uncalibrated model score with a heuristic and leaves omitted candidates on their old scale. Prefer separate `deterministic_confidence`, `model_rank` and evidence fields, or calibrate on labeled tasks.
- Replacing descriptions lets a cheap model change what the caller reads even when the rank does not change. Keep the deterministic description and add the model's interpretation separately.
- Run three arms on a labeled set of failed tasks: current; calling agent ranks from candidates plus timeline evidence; and deterministic only. Add a fourth arm with a short failure-taxonomy document. Score top@1, candidate recall, valid event IDs, tokens, latency and whether the agent's next action improved, over repeated runs.
- Prior, before that test: deleting the cheap call is attractive if the agent is already on a stronger model, the timeline is about 3,000 tokens and the deterministic candidates are good. Keeping it is defensible if replay shows fixed ranking changes by task type, as in the paper. The call-time cost matters less than whether it changes the agent's diagnosis.

### Gaps
- I found no paper on LLM re-ranking of deterministic candidates from agent execution traces, and no direct test of the calling-agent-ranks option.
- No source supports a particular blend weight. Calibration of cheaper models for this task is unknown.

## When should these jobs be separate model calls, steps in the agent loop, or deterministic code?

### Takeaway
Use code for facts and mechanical state, a loop step when the work needs the agent's tools or the model's full context, and a separate call only when isolation, timing outside a turn, or a verified cost or latency advantage requires it. Each separate call needs its own baseline, failure mode and measurement.

### Cited Findings
- Add complexity only when it demonstrably improves outcomes — [Anthropic, Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- Deterministic trimming avoids added model calls and summary drift, and deterministic preprocessing can remove most log context — [OpenAI cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/session_memory); [Claude Code, costs](https://code.claude.com/docs/en/costs)
- Mechanical extraction retained far more recurring details than LLM summaries, but retaining details did not ensure continuation success — [Compact-Bench](https://arxiv.org/html/2610.31710)
- A separate classifier can be justified by isolation: auto mode hides tool outputs and agent reasoning to limit manipulation and injection exposure — [Anthropic, Claude Code auto mode](https://www.anthropic.com/engineering/claude-code-auto-mode)
- Side calls using a different system prompt, tools or model lose prefix-cache reuse, while forks keeping the same prefix retain it — [Claude Code team, Prompt caching is everything](https://claude.dev/blog/lessons-from-building-claude-code-prompt-caching-is-everything/); [Claude Code, prompt caching](https://code.claude.com/docs/en/prompt-caching)
- OpenAI's baseline-first method applies to side calls: establish a strong-model baseline, then downgrade while checking results — [OpenAI, A practical guide to building agents (PDF)](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)

### Inferences

| Component | Evidence-based position | Main test |
|---|---|---|
| Relationship router | Keep for now because of the local A/B. It is not proven better than always-available history. | Add a router-off arm with the tool-history note, and report error by direction, latency and deadline hits. |
| Summarizer | Separate background pass is reasonable. Fast-model default is unproven. Add deterministic state. | Paired full-history vs summary continuation, fast vs primary model, with retrieval, repeated work and cache reuse. |
| Background interpreter | Weakest case for a model call once the small-model premise is rechecked. | Template vs model sentence vs agent re-entry on hosted models, scoring wrong status and re-runs. |
| Diagnose ranking | Evaluate. Remove the description overwrite and the unsupported blend before judging it. | Labeled replay of current, agent-ranks and deterministic arms, including candidate recall. |

- Every side call should record a named purpose, model, latency, token use, cache use, outcome and fallback taken, so that "the call succeeded" is separate from "the call helped".
- For this project, N-of-M runs on realistic scenarios are the right unit, and completion alone should not be the deciding metric.

### Gaps
- No source gives a general quantitative threshold for when a side call pays for itself. That depends on aidaemon's own models, billing path and cache behavior, and needs local measurement.
