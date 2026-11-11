# Agent memory formation and procedural learning by background model calls (state as of 2026-11-09)

Reading guide. "Vendor" means the number is self-reported by the party selling the system. Older material is dated inline. Most 2026 papers below are single-group arXiv preprints without peer review or replication. Aidaemon figures come from the assignment brief: 12 write events / 226 calls = 5.3% yield; 187k input tokens / 226 = about 830 tokens per call and about 15.6k input tokens per write (my arithmetic).

## Which memory should be written in the hot path versus in the background?

### Takeaway
The field has converged on a split rather than on either pole: the working agent writes what it deliberately decides is worth keeping, and a later offline pass verifies, deduplicates and reorganizes. Claude's products and Letta add the offline pass on top of agent-authored notes, with no independent per-turn extractor. Mem0 and AgentCore extract automatically outside the agent, ChatGPT synthesizes in the background from chat history at an undisclosed cadence, and GitHub rejected an offline pipeline outright.

### Cited Findings
System by system:
- **LangMem/LangGraph:** offers both. In the hot path the agent "consciously" saves through tools; in the background, memories are "subconsciously" extracted and the agent keeps responding — [LangMem background guide](https://langchain-ai.github.io/langmem/background_quickstart), [hot path guide](https://langchain-ai.github.io/langmem/hot_path_quickstart). Stated trade-off: hot path gives immediate availability and lets users be told, but adds latency, complexity and divides the agent's attention; background avoids this, but timing is the hard part, because too-infrequent writes leave other threads without context — [LangChain memory concepts](https://docs.langchain.com/oss/python/concepts/memory).
- **Mem0:** asynchronous automatic extraction outside the agent. In April 2026 it moved from two LLM passes (extract, then ADD/UPDATE/DELETE reconcile) to one ADD-only call, arguing that reconciliation destroyed information. Vendor: extraction latency "roughly halved," with no method given — [Mem0 algorithm post](https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm).
- **Letta:** Letta Code's "dreaming" uses background subagents, triggered every N steps, on compaction, or never; they review conversations and update memory. An optional "Agent reviews before applying" adds a second background reviewer, not human approval — [Letta docs](https://docs.letta.com/guides/agents/architectures/sleeptime). Letta distinguishes these dream subagents from server-side sleep-time agents — [Letta memory docs](https://docs.letta.com/letta-code/memory).
- **Zep/Graphiti (Jan 2025 paper, older):** builds a temporally aware knowledge graph from ongoing conversation and business data — [Zep paper](https://arxiv.org/abs/2501.13956). Graphiti's update mechanism is soft revocation through validity timestamps — [Revocation study](https://arxiv.org/pdf/2609.08258).
- **ChatGPT:** "Dreaming," announced June 4, 2026, automatically updates memory from past chats in the background, with legacy Saved Memories as a fallback option. Later controls let users delete items shown in the memory summary. OpenAI credits about a 5x reduction in serving compute with making a Free rollout possible — [OpenAI post](https://openai.com/index/chatgpt-memory-dreaming/), [release notes](https://help.openai.com/en/articles/6825453-chatgpt-release-notes). Both pages returned 403 to my fetcher, so these details come from search extracts of those pages. Power users complain about lost control and want pins and revision history — [community thread](https://community.openai.com/t/dreaming-v3-sacrifices-real-personalization-for-compute-savings-a-power-user-s-perspective/1383010) (anecdotal).
- **Claude memory tool:** purely hot path and model-driven. Claude views the directory first and writes as it works; storage is client-side. Anthropic advises implementers to "periodically delete memory files that haven't been accessed in a long time" — [memory tool docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool).
- **Claude Managed Agents:** agents write during work; Anthropic says these writes are "local and incremental," so stores accumulate "duplicates, contradictions, and stale entries." A Dream reads a store plus 1–100 sessions and produces a separate output store; the input is never modified, so the output can be reviewed or discarded. Research preview; cost scales roughly linearly with sessions — [Dreams docs](https://platform.claude.com/docs/en/managed-agents/dreams).
- **Claude Code:** auto memory is written by Claude as it works in the session. The first 200 lines or 25KB of `MEMORY.md` load at start; topic files load on demand — [Claude Code memory docs](https://code.claude.com/docs/en/memory). That page did not mention Auto Dream when I fetched it. Descriptions of its triggers (for example 24 hours plus 5 sessions) are only third-party and disagree — [SFEIR](https://institute.sfeir.com/en/articles/claude-code-dream-auto-dream-memory-consolidation/), [SupaLaunch](https://supalaunch.com/blog/claude-code-dreams-auto-dream-memory-consolidation-guide).
- **GitHub Copilot (contradicting evidence for offline curation):** agents create memories through a tool call, with citations to code. GitHub considered an offline service to dedupe, resolve conflicts and expire, and judged it too complex and costly at its scale and still in need of read-time reconciliation. It chose read-time citation verification instead — [GitHub engineering](https://github.blog/ai-and-ml/github-copilot/building-an-agentic-memory-system-for-github-copilot/).
- **AWS AgentCore:** background extraction after events land, then an LLM reconciles against similar memories. If consolidation ultimately fails, the memory is still stored, favoring retention over correctness — [AWS deep dive](https://aws.amazon.com/blogs/machine-learning/building-smarter-ai-agents-agentcore-long-term-memory-deep-dive).

Evidence on the split:
- Anthropic's November 6, 2026 study: writing notes during a task competes with the task, while dreaming can run later but its outputs are not immediately available. In a Pokémon test, a run with an uncorrected note never reached Cerulean City, and a run with a dreaming-corrected note did so by action 1,935. On CL-Bench's database task with memory, mean reward moved from 0.564 to 0.828 on Opus 5 and 0.655 to 0.838 on Opus 5.5 (vendor) — [Anthropic](https://www.anthropic.com/research/components-of-agent-memory).
- ProMem's critique: background summarization is "ahead-of-time" because the extractor does not know future tasks, and it is one-off with no verification, so errors persist — [ProMem](https://arxiv.org/pdf/2601.04463).

### Inferences
- Aidaemon's stack is heavier on independent background extractors than any product I reviewed, and I found no system running this many in parallel. The "default" comparison is one agent-authored path plus one consolidator.
- Candidates for hot path: explicit "remember" and clear corrections. Candidates for background: anything needing cross-session comparison (fact consolidation, session analysis, procedure mining). Progressive extraction is the component with the weakest case in the product survey, but Mem0 and AgentCore are real counterexamples, so it should be tested rather than deleted by analogy.
- GitHub's verifiable-citation design depends on code being a checkable oracle. A personal agent has that oracle for files and tool receipts, not for statements about the user's life, so GitHub's rejection of offline curation does not transfer wholesale.

### Gaps
- No head-to-head study isolates hot-path versus background writing with the same models and tasks.
- ChatGPT's cadence and pipeline internals are undisclosed; the OpenAI pages could not be read directly.
- I could not confirm which model Letta's background agents default to.

## Per-turn versus batched or session-end extraction: quality, cost, yield, and whether a low write rate is healthy

### Takeaway
Evidence favors session or batch granularity on cost and shows no measured quality penalty, but the studies are few and confounded. A low write rate is neither good nor bad on its face. Several systems achieve about 90% precision with under 60% recall, so low yield can mean a discerning extractor or a blind one; only a recall audit and downstream-use measures can separate them.

### Cited Findings
- Session-granularity curation (one LLM pass per session, offline consolidation) scored 0.902 on LoCoMo with gpt-4.1-mini, against 0.913 for full context, using 87% fewer tokens. The authors explicitly contrast this with per-turn systems such as Mem0 and A-Mem and suggest consolidation "every 10-20 sessions." LongMemEval-S 85.5% against full context 65.6% — [Curation and Consolidation, Oct 2026](https://arxiv.org/html/2610.39790v1).
- Consolidation in that paper bought compression at a small accuracy cost: baseline 0.902; min cluster size 5 gives 21.7% compression at 0.896; size 2 gives 30.9% at 0.876 — [same](https://arxiv.org/html/2610.39790v1).
- Against consolidating at write time: Mem0 argues that update and delete operations lost details and that keeping all facts preserves state history (vendor). Its own contradiction-resolution scores remain among the weakest categories: 35.7 at 1M and 32.5 at 10M on BEAM — [Mem0](https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm).
- Extraction quality on HaluMem: Mem0 integrity (recall) 42.91, accuracy 86.26, 10.02K tokens; Memobase 14.55 / 92.24; LightMem about 58 / 92; ProMem 81.37 / 94.68 at 11.59K tokens. ProMem adds completeness checks and turn-level entailment verification, and argues memory is write-once-read-many and that errors cost more than tokens — [ProMem](https://arxiv.org/pdf/2601.04463). Author-run benchmark; ProMem's stack mixes models.
- Write-time cost measured independently: GPT-5-nano extraction cost $0.0435 per 101.6k-token conversation, with memory cheaper than long-context from about 10 turns (9–13 depending on context size). Accuracy was much lower for the memory system, however: LongMemEval 49.0% against 82.4%; LoCoMo 57.68 against 92.85 — [Cost-performance study](https://arxiv.org/html/2603.04814v1). It tests only Mem0's flat design.
- Selective writing: at 8:1 distractor ratios, write-time gating held accuracy while read-time filtering collapsed to 0 — [Zahn & Chana](https://arxiv.org/html/2603.15994). A dual-layer design cut storage ratio from 64.81% to 32.08% while closely matching exhaustive-memory performance — [Dual-layer memory](https://arxiv.org/html/2608.22215v2). Different benchmarks with no shared baseline.
- A survey's verdict: "Storing every interaction verbatim is tempting and almost always wrong"; the right threshold depends on the cost of misses, since a medical agent cannot miss an allergy — [Memory survey](https://arxiv.org/html/2603.07670v1).
- Contradicting a strict gate: Anthropic found that fixed capture rules hurt. "Record only failures" reduced the memory gain by 0.234 on Opus 5 and 0.150 on Opus 4.7. Its advice is to save salient, non-obvious strategies and verified rules, and skip volatile state. "Tell Claude what memory is for, but not how to use it" (vendor) — [Anthropic](https://www.anthropic.com/research/components-of-agent-memory).
- Too much memory also hurts: Sonnet 3.7 accumulated 413 files and 978K characters and spent steps reorganizing them, earning one badge; Fable 5 kept 10 files and 43K characters and earned four — [Anthropic](https://www.anthropic.com/research/components-of-agent-memory). Model capability is confounded here.

### Inferences
- A 5.3% yield at 830 tokens per call is cheap in absolute terms. Cost per write (15.6k input tokens) is not a decisive argument for deletion on a fast tier; the stronger arguments are duplicate writes, poisoning surface and latency of learning.
- Proposed yield measures, in priority order: (1) recall audit: hand-label N exchanges for durable facts and compute extractor recall/precision; (2) marginal yield: facts this pipeline found that no other pipeline found within a window; (3) use rate: writes retrieved and rendered within 30 days; (4) counterfactual benefit by ablation; (5) correction or retraction rate; (6) tokens per surviving, used fact. Raw writes/calls measures none of these.
- A batch design for the same exchanges (session-end pass over all eligible exchanges) would cut call count sharply. The test is whether facts arrive too late for same-day cross-session use, which is LangChain's stated background risk.

### Gaps
- I found no published production base rate for how many turns should yield a durable fact. The 5.3% cannot be benchmarked externally.
- No controlled study directly compares per-turn and session-end extraction with extractor, model and corpus held constant.

## Multiple overlapping extraction pipelines: duplicates, contradictions, poisoning, stale facts; provenance and write gates

### Takeaway
Automatic and aggressive write channels measurably raise poisoning exposure, and content screening cannot detect plainly false statements. The defensible controls are narrow write scope, origin binding with restricted influence at retrieval, enforced revocation, and verification against an external source where one exists. Provenance as a soft ranking weight has failed in one careful test.

### Cited Findings
Poisoning and write channels:
- MPBench separates four memory write channels including compaction-driven writes; "agents designed to write and retrieve memory more aggressively are more exploitable." Average write-attack success was 66.67% on Hermes against 34.25% on OpenClaw, and average influence in a later session 64.70% against 17.40%. Salience-driven compaction poisoning reached 85.17% on Hermes. Authors' first recommendation: "precise, scope-limited write policies" — [From Untrusted Input to Trusted Memory, Jun 2026](https://arxiv.org/pdf/2606.04329).
- False statements defeat screening: 1.2% poisoned corpus cut LongMemEval accuracy from 0.850 to 0.300, and a four-stage content screen with 0.832 prompt-injection recall refused 0 of 360 poisoned memories. The shipped provenance ranking weight was indistinguishable from no defense (p=0.80). A larger weight recovered utility by effectively excluding untrusted content, and accuracy fell to 0.0417 when real evidence itself arrived through untrusted channels. Authors propose bounded occupancy plus write-time origin binding, and state the proposal is unevaluated; the attacker is non-adaptive — [Utility Under Attack, Aug 2026](https://arxiv.org/pdf/2608.21230).
- MINJA (query-only injection) is commonly cited at over 95% injection and 70% attack success under idealized conditions; I took this from search-result summaries and did not read the original paper — [topic summary](https://www.emergentmind.com/topics/persistent-memory-poisoning). In a January 2026 replication with 6 pre-existing legitimate memories, GPT-4o-mini attack success fell from 62% to 6.67%, then rose to 38% when retrieving 10 memories rather than 3. Trust-score filtering performed poorly: it rejected all candidates on one model, and on another accepted 82 entries with trust 1.0, of which 54 were poisoned — [Memory Poisoning Attack and Defense](https://arxiv.org/html/2601.05504v2). Retrieval depth is therefore a security parameter.
- Write-time defenses suppress direct attacks but not compositional or dormant ones, and the authors argue for periodic lifecycle cleaning — [MemPoison, Jul 2026](https://www.alphaxiv.org/abs/2607.14651). The abstract-level claim was all I read.

Staleness and contradictions:
- A study of five memory systems (Graphiti, Zep CE, Mem0, LangMem, Cognee) concludes none enforces soft revocation by default. Its table is more nuanced: Graphiti and Zep CE return revoked records by default, Mem0 withholds them unless a flag overrides, and LangMem and Cognee retain no revoked record at all because they overwrite or re-ingest. Where a revocation label was visible to retrieval, the revoked record was returned and outranked its replacement. Unsafe action occurred in 699 of 1,620 trials (43.1%) on the two exposed systems, ranging 15–60% per model, with no correlation to capability tier. Prompt hardening alone only reduced this to 37.2%. Mem0's measured consolidation path added a new fact without retiring the old one — [Revoked but Still Authoritative, Sep 2026](https://arxiv.org/pdf/2609.08258).
- GitHub's read-time design: seeded false memories with bad citations were caught and corrected, and the pool "self-healed" (no numbers published). A memory's freshness only resets when validated, not when retrieved; default expiry is now 90 days, configurable 7–180 for repository facts — [GitHub blog](https://github.blog/ai-and-ml/github-copilot/building-an-agentic-memory-system-for-github-copilot/), [Copilot docs](https://docs.github.com/en/copilot/concepts/agents/copilot-memory). Earlier secondary reports of 28 days appear outdated.

Provenance and gates in the literature:
- Recommended write path: filter low-signal records; normalize; merge duplicates; tag timestamp, source, confidence; rank user statements over agent inferences — [Memory survey](https://arxiv.org/html/2603.07670v1). Mem0's new algorithm does the opposite on one point, storing agent-generated statements with the same weight as user statements — [Mem0](https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm).
- Verification at atomic-fact level against the source turn: keep on entailment, otherwise revise or discard — [ProMem](https://arxiv.org/pdf/2601.04463).
- Evidence-gated writing on utility, consistency and evidence sufficiency — [EGMW](https://www.researchgate.net/publication/406632867_Evidence-Gated_Memory_Writing_for_Personalized_Large_Language_Model_Agents) (seen only through a search abstract).
- Signed-entry tooling exists but its claims are unvalidated — [mguard](https://github.com/mguard-ai/mguard).

### Inferences
- Each extra automatic channel is an additional poisoning route, so a pipeline with near-zero use should have to justify its security surface. This argues for removing or disabling the unused fact-extraction backend unless it wins an evaluation.
- Task learning reads an activity log that can contain untrusted tool output. Test directly that aidaemon's taint rules reach all seven pipelines, including task learning and skill promotion, with a canary memory attack through each channel.
- Test revocation at the render boundary, not just in storage: load an old fact and its replacement, and assert the old one cannot reach the prompt as current.
- Treat provenance as origin binding: which turn, role, tool, trust class, pipeline ID. Do not assume a confidence score or additive trust term provides safety.

### Gaps
- I found no study that measures the rate of duplicates or contradictions specifically created by running several extractors over the same history. The risk is inferred from single-pipeline failure modes.
- Adaptive attackers against provenance gates are not yet measured; origin-bound provenance has no independent evaluation.

## Which model tier do these jobs need?

### Takeaway
For fact extraction and session-level curation, mid-size fast models appear close to frontier on available benchmarks, though the oldest small models lose noticeable accuracy. For procedure and skill writing, evidence on tier is thin and mixed; there is no clean test of a strong writer versus a fast writer holding the pipeline constant.

### Cited Findings
Fact extraction and curation:
- LoCoMo overall: gpt-5.4 0.906; gpt-4.1-mini 0.902; gpt-4o-mini 0.835; Qwen3-8B 0.762. Authors call gpt-4.1-mini comparable to gpt-5.4 and note the 6.7-point loss on gpt-4o-mini as a limitation — [Curation and Consolidation](https://arxiv.org/html/2610.39790v1). The full system backbone changes, so the writer is not isolated.
- ProMem reports Llama3.1-8B-based extraction beats Mem0 on memory integrity, with QA gains; its pipeline also uses a stronger hosted model for part of verification — [ProMem](https://arxiv.org/pdf/2601.04463).
- Extraction with GPT-5-nano at low effort gave the worst memory-versus-long-context gaps in one study, but the reader, architecture and retrieval are confounded — [Cost-performance study](https://arxiv.org/html/2603.04814v1).

Consolidation and procedures:
- Anthropic's Dreams supports Sonnet 4.6 and up, with no Haiku on the list, even though the memory tool itself runs on Haiku. Anthropic gives no reason — [Dreams](https://platform.claude.com/docs/en/managed-agents/dreams), [memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool).
- Anthropic: newer models "closed the learning loop" better, and Opus 5 was the first to consistently record rules, evidence and traps. Weak readers matter too: Sonnet 4.6 gained 0.330 to 0.364 unprompted and 0.641 when told to read memory and test theories (vendor) — [Anthropic](https://www.anthropic.com/research/components-of-agent-memory). This measures models using memory in the main loop, not side-call writers.
- Strong writer helping a weak consumer: GPT-5-mini teacher memory gave 4B–8B students +27.2 points on AppWorld, +11.2 on BFCL V3, +3.4 on ToolSandbox — [Agent Memory Distillation](https://arxiv.org/abs/2608.07169). GPT-4o procedures used by Qwen2.5-14B added about 5% task completion on TravelPlanner — [Mem^p](https://arxiv.org/html/2508.06433v2).
- Skill generation by backbone: EvoSkill ranged from 10.2% (Kimi K2.5) to 20.3% (GPT-5), while SkillSeekers stayed at 14.4–20.9%. So backbone sensitivity depends on the pipeline — [SkillGenBench](https://arxiv.org/html/2605.18693v1).
- Self-written skills did not improve with model strength in SkillsBench: Opus 4.6 +1.4 points, Opus 4.5 −0.4, Sonnet 4.5 −2.1, Haiku 4.5 0.0, GPT-5.2 −5.6 — [SkillsBench](https://arxiv.org/html/2602.12670v3). Task-conditioned self-generation only.
- Hermes's skill Curator defaults to the main chat model, with a configurable cheaper slot; a sweep takes 50–100 API calls — [Hermes Curator](https://hermes-agent.nousresearch.com/docs/user-guide/features/curator).
- ACE's Reflector is a reported dependency: poor insight extraction gives noisy or harmful updates, per a secondary summary — [ACE review](https://www.themoonlight.io/review/agentic-context-engineering-evolving-contexts-for-self-improving-language-models). The open-source example runs DeepSeek-V3.1 in all roles — [ACE repo](https://github.com/ace-agent/ace). ACE's abstract reports gains using a smaller open-source model — [ACE](https://arxiv.org/abs/2510.04618).

### Inferences
- Fast-tier extraction is defensible for facts, provided the actual fast model is at least roughly gpt-4.1-mini class. Pin and re-evaluate on model change.
- Procedure enhancement on the strongest model is unproven both ways. Its volume should be low, so cost is probably not the deciding factor; a paired test (same procedures, strong versus fast writer, blind execution on replayed tasks) would settle it cheaply. If there is no measurable gain, drop the tier, and if the skill's value itself is unproven, the tier is moot.
- Skill promotion on the fast tier, by contrast, is the least supported tier choice: it is the most consequential write, performed rarely, by a cheap model, with no execution check.

### Gaps
- No controlled test holds pipeline and reader fixed while varying only the writer or consolidator.
- No published comparison for Anthropic's current Haiku/Sonnet/Opus tiers as memory side-call writers.

## Procedural memory and self-improving agents: does automated skill creation help, is it harmful or ignored, and how is success attributed?

### Takeaway
Procedural learning clearly helps when it is grounded in executable verification or repeated environment feedback and evaluated on the same kind of task. Generic text skills written from little evidence show roughly zero benefit, and libraries distilled for unknown future tasks can fall below no-skill baselines. Success counts alone do not attribute outcomes to a procedure; only with/without comparisons do, and few systems run them.

### Cited Findings
Classic work (2023–2025, older, mostly controlled environments):
- Voyager (2023): executable skill library in Minecraft, 3.3x more unique items and milestones up to 15.3x faster, using environment feedback and execution errors — [Voyager](https://arxiv.org/abs/2305.16291).
- Reflexion (2023): verbal self-reflection in episodic memory across trials; 91% on HumanEval against 80% for GPT-4 — [Reflexion](https://arxiv.org/abs/2303.11366).
- ExpeL (2023): extracts natural-language insights from experience with improving results and transfer; abstract gives no numbers — [ExpeL](https://arxiv.org/abs/2308.10144).
- Dynamic Cheatsheet (Apr 2025): reuse of validated code raised GPT-4o Game of 24 from 10% to 99%; only about 8–9% on knowledge benchmarks (GPQA, MMLU-Pro) — [Dynamic Cheatsheet](https://arxiv.org/abs/2504.07952). The large gains are concentrated where a reusable program exists.
- ACE (Oct 2025): itemized delta updates; +10.6% on agent benchmarks and +8.6% in finance — [ACE](https://arxiv.org/abs/2510.04618). A secondary source reports a Dynamic Cheatsheet rewrite collapsing a context from 18,282 tokens to 122, with accuracy falling from 66.7% to 57.1% — [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2026/09/agentic-context-engineering/). Whole-document rewriting has a known failure mode that bears on procedure enhancement.
- Mem^p: proceduralized memory beat scripts and raw trajectories (ALFWorld test 77.86 against 42.14 with no memory) — [Mem^p](https://arxiv.org/html/2508.06433v2).

Automated skills under evaluation:
- SkillsBench (Feb 2026; 84 tasks): curated skills +16.2 points on average, with 16 of 84 tasks negative; self-generated about −1.3 (the table's mean row gives −3.3, unreconciled in the paper). Four or more skills +5.9 against +17.8 for one; "comprehensive" skills −2.9. Length confound is acknowledged — [SkillsBench](https://arxiv.org/html/2602.12670v3).
- SkillGenBench critique of SkillsBench: its self-generation condition is task-local. Across five pipelines, only one exceeded the no-skill baseline on average (+0.6 Code, +1.6 Doc); differences are within about ±5 points CI. Task-agnostic distillation can fall below no-skill baseline because of negative transfer. Failures concentrate in interface/schema errors and wrong assumptions — [SkillGenBench, May 2026](https://arxiv.org/html/2605.18693v1).
- MUSE-Autoskill (positive result): skills distilled from each agent's successful trajectory, packaged with tests, and registered only if tests pass. Accuracy 60.35% against 53.19% no-skill and 68.40% human. On the 35 tasks with a skill, 76.9% rose to 87.94%. 16 of 51 tasks yielded no skill. Three tasks regressed, including hvac-control 80% to 20% due to a trajectory-specific routine. Re-testing on the same task may overstate benefit, and each skill comes from one trajectory. Authors estimate generation pays back after about three reuses — [MUSE-Autoskill](https://arxiv.org/html/2605.27366v1).
- Hermes: agent-created skills, with a Curator using deterministic usage-based lifecycle by default (stale at 14 days, archive at 30, in the config defaults; the same docs elsewhere say 30/90). LLM consolidation is opt-in. Telemetry tracks views, uses and patches, not outcomes — [Hermes Curator](https://hermes-agent.nousresearch.com/docs/user-guide/features/curator). One independent trial: one task, one run, consolidation only dry-run, later benefit untested — [CodeCut](https://codecut.ai/hermes-autonomous-skills-curator/).
- Anthropic's own authoring guidance: build evaluations before writing skills, establish a baseline without the skill, use a fresh instance to test, and watch for content Claude ignores — [Skill best practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices).
- Stale procedure risk: a narrow qwen3:8b study found no interference when current-task evidence was explicit and sufficient, and warns against generalizing — [Procedural Memory Under Change, Sep 2026](https://arxiv.org/abs/2609.09774).

Poisoned learning:
- PoisonedEvolution: 10% attacker-supplied records embedded target behavior in 91.0% of SkillClaw trials (546/600) and 61.5% of Trace2Skill. Three consistent records in a batch of 30 were sufficient. Recurrence and causal framing were the key factors, and the authors call evidence promotion a security boundary — [Trajectory Poisoning, Aug 2026](https://arxiv.org/abs/2608.05563). Related: query-only backdoors planted through consolidation — [Query-only backdoors](https://arxiv.org/pdf/2608.08303); malicious skills activating through retrieval — [EvoSkill Injection](https://arxiv.org/html/2608.30429v1).

### Inferences
- The "≥5 successes, ≥80%" gate measures recurrence, not causation. A procedure can pass by describing tasks the model would solve anyway. It is also the kind of recurring evidence that trajectory-poisoning attacks exploit.
- The strongest available promotion design is MUSE's: require an executable or replayable check, compare with and without on held-out tasks, and hold back any skill that produces a regression. Drafts lacking a check should be labeled as unverified notes.
- Procedure enhancement should be evaluated against the raw procedure, since rewriting can remove the specific detail that made it useful. Compare length-matched text too, following SkillsBench's own control suggestion.
- Keep the active library small; SkillsBench's +5.9 at 4+ skills warns against bulk-activating the 19 drafts, even if each is sound.
- Task learning from a 4 KB log is a narrow, single-trajectory source. MUSE's hvac regression is the warning; MUSE's coverage loss shows a verifier also rejects a lot, and that rejection is acceptable.

### Gaps
- I found no study measuring how often auto-created skills are invoked in the wild, and no production A/B of auto-generated skills.
- No published attribution method for skills in a personal-assistant setting, where tasks rarely repeat and outcomes lack ground truth.
- Reflexion, Voyager and ExpeL details beyond abstracts were not re-read.

## Human review queues for automated learning: what happens when nobody reviews?

### Takeaway
The direct evidence is sparse, but product designs point the same way: nearly all have moved from pre-approval to automatic application plus visibility, audit, expiry and rollback. Queues with no reviewer are a known failure and offer no safety. An unreviewed draft should either expire or pass an automated evaluation; it should not sit indefinitely or become active by default.

### Cited Findings
- Hermes makes pre-landing approval opt-in (`skills.write_approval`). Its defaults are an append-only write ledger with per-entry rollback, run reports, nothing auto-deleted, pin and lock controls, and usage-driven archival — [Hermes Curator](https://hermes-agent.nousresearch.com/docs/user-guide/features/curator). The author of an independent test still argues merge suggestions need checking first — [CodeCut](https://codecut.ai/hermes-autonomous-skills-curator/).
- Letta's optional review pass is another agent; the docs state it does not ask for user approval — [Letta docs](https://docs.letta.com/guides/agents/architectures/sleeptime).
- Claude Dreams uses a staged-output design: a new store is produced and the input untouched, and the user adopts or discards it — [Dreams](https://platform.claude.com/docs/en/managed-agents/dreams).
- ChatGPT moved to automatic updates while adding summary-level controls and memory sources, which OpenAI says may not show every factor influencing an answer — [release notes](https://help.openai.com/en/articles/6825453-chatgpt-release-notes), [GPT-5.5 Instant post](https://openai.com/index/gpt-5-5-instant/) (search extracts only).
- Copilot needs no approval: facts are validated at use, reinforced by "upvote," dropped after enough downvotes, and expire after 90 days without reinforcement; users can review and delete — [Copilot docs](https://docs.github.com/en/copilot/concepts/agents/copilot-memory).
- Cursor removed its Memories management UI in 2.1.x, leaving memories invisible or unclear; staff statements conflict on whether the feature itself was removed. Users complained about being unable to see or clear them — [Cursor forum](https://forum.cursor.com/t/memories-not-showing/143820), [Can't clear memories](https://forum.cursor.com/t/cant-clear-memories/148254). An earlier third-party guide says background memories needed approval and a dismissed prompt meant the memory was never saved — [localskills](https://localskills.sh/blog/cursor-memories-guide) (unconfirmed).
- General human-oversight sources describe rubber-stamping after long runs of correct proposals, and quality degrading as queues grow — [ProdPad](https://www.prodpad.com/glossary/human-in-the-loop/). Practitioner guides in the same search recommend risk-based sampling, tracking queue depth and timeouts, and even seeding known-wrong items (for example [Ready for Role](https://readyforrole.com/blog/human-in-the-loop-design.html), [Olostep](https://www.olostep.com/blog/human-in-the-loop)); I did not verify which guide makes which claim, and none is empirical. A 2026 paper lists human fatigue under backlog as future work rather than a measured result — [Operating Imperfect AI](https://arxiv.org/pdf/2601.22295).

### Inferences
- 19 pending, 0 reviewed is a broken loop. The queue is producing token cost and no learning, and its safety value is nominal because nothing reaches production. That is safe but useless.
- Recommended order of options to test: (1) stop generating while the backlog exceeds a small cap; (2) digest notification with at most 3 drafts, each showing evidence and an approve/discard action; (3) expiry after a set window with a tombstone, so the same procedure is not redrafted without new evidence; (4) an evaluated path for drafts that pass replay. Pure auto-approve on the existing success counts is not supported, given the trajectory-poisoning and SkillsBench results.
- Record reviewer inaction as data. If nobody reviews after a fixed trial period, that is direct evidence that this surface should be auto-evaluated or removed.
- Apply least-privilege activation for any auto-promoted skill: scope, rollback, and a probation period where regressions demote it.

### Gaps
- I found no empirical study of auto-expiry, abandonment rates, or approval rates for agent-learned content, and no published timeout defaults.
- Notification effectiveness for memory or skill review has no measured data that I could find.

## What should benchmarks and their critics make us measure, and what evaluation should exist before keeping or deleting a pipeline?

### Takeaway
Public recall benchmarks cannot justify a personal-agent pipeline: LoCoMo is contested and near saturation, and vendor numbers conflict. The right bar is per-pipeline ablation on the system's own replayed tasks and memory-operation regression cases, reporting task outcomes, memory quality, cost and security together over repeated runs.

### Cited Findings
Disputes and limits:
- Zep vs Mem0: Zep claimed 84% on LoCoMo; Mem0 re-ran it at 58.44%, alleging inclusion of adversarial questions; Zep corrected to 75.14% and alleges Mem0 misconfigured Zep (user roles, timestamps, sequential search) — [Zep](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/), [Atlan summary](https://atlan.com/know/zep-vs-mem0/). Unresolved; Zep also argues transcripts fit within a single context window.
- Mem0 now reports LoCoMo 92.5 and LongMemEval 94.4 (vendor, managed platform, single retrieval call), and itself says those benchmarks can be inflated by larger windows and stronger models. BEAM drops to 64.1 at 1M and 48.6 at 10M, with temporal reasoning at 16.3 — [Mem0](https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm).
- Letta (vendor): a plain filesystem agent on gpt-4o-mini scored 74.0% on LoCoMo, above Mem0's then-reported 68.5% — [Letta](https://www.letta.com/blog/benchmarking-ai-agent-memory).
- Independent-ish: long-context GPT-5-mini beat a Mem0-style system by 35.2 points on LoCoMo and 33.4 on LongMemEval, but only 7.3 on PersonaMem v2 — [Cost-performance study](https://arxiv.org/html/2603.04814v1). A full-context baseline should always be included where feasible.
- A paper flags a refusal-matching bug in the reference runner in Mem0's codebase and reports practitioners failing to reproduce LoCoMo numbers — [MemMachine](https://arxiv.org/pdf/2604.04853) (itself a competing system; seen via search extract). MemPalace's 100% LongMemEval reportedly needed several iterations of LLM reranking — [MemPalace critique](https://arxiv.org/pdf/2604.21284).
- Models near-perfect on LoCoMo fall to 40–60% on MemoryArena's interdependent multi-session tasks, and a long-context-only baseline falls from above 80% to about 45% — [Memory survey](https://arxiv.org/html/2603.07670v1).
- LongMemEval-V2 tests agent experience over up to 115M tokens: AgentRunbook-C 72.5%, off-the-shelf coding agent 69.3%, best RAG 48.5% — [LongMemEval-V2](https://arxiv.org/abs/2605.12493). An off-the-shelf agent nearly matching the best system is a warning for custom pipelines.
- Locomo-Plus targets implicit-constraint recall that surface-form tests miss — [Locomo-Plus](https://arxiv.org/html/2602.10715v1).

Better evaluation patterns:
- GitHub's production A/B: coding-agent PR merge rate 83% to 90%, positive code-review feedback 75% to 77%, offline precision +3% and recall +4% (vendor, p<0.00001 stated) — [GitHub](https://github.blog/ai-and-ml/github-copilot/building-an-agentic-memory-system-for-github-copilot/). Outcome metrics, but aidaemon has far too little traffic to replicate this.
- Anthropic's design: stateless baseline vs memory on sequential fresh sessions, by model, varying what gets saved and how it is prompted — [Anthropic](https://www.anthropic.com/research/components-of-agent-memory).
- Survey's recommended stack: task effectiveness; memory quality (retrieved precision/recall, contradiction rate, staleness distribution, coverage); efficiency (tokens, latency, storage growth); governance (leakage, deletion compliance, access scope). Ablate write policy, retrieval and compression separately, keep ground-truth regression tests, and log every memory operation — [Memory survey](https://arxiv.org/html/2603.07670v1).
- Security cases belong in memory evaluation: poisoned-corpus utility loss — [Utility Under Attack](https://arxiv.org/pdf/2608.21230); persistence across sessions — [MPBench](https://arxiv.org/pdf/2606.04329); revocation at retrieval — [Revocation study](https://arxiv.org/pdf/2609.08258).

### Inferences
Proposed evaluation for the seven aidaemon components (design proposals, not sourced results; use synthetic data in the repo):

| Component | Minimum test before keep/delete | Deletion signal |
|---|---|---|
| Progressive extraction | Labeled-exchange recall/precision; replay week of history with and without; compare to a session-end batch | Same facts found by batch or consolidation within acceptable delay |
| Fact extraction | Run once on the same replay as a challenger | No recorded use, no marginal facts: remove surface |
| Fact consolidation | Contradiction/staleness cases; old-versus-new render test; accuracy with and without | Compression gain without improved answers, or any lost true fact above a threshold |
| Session analysis | Does an episode change later answers or goal follow-through versus raw transcript search? | Episodes never retrieved; goals mostly duplicate |
| Task learning | Canary injection through the activity log; held-out similar-task replay | Facts/procedures never reused or high correction rate |
| Procedure enhancement | Blind paired runs: raw versus enhanced, strong versus fast writer, length-matched | No win over raw procedure |
| Skill promotion | Replay check or executable test; with/without on held-out tasks; backlog and review metrics | No draft beats baseline, or no review or auto-evaluation path exists |

- Report N-of-M runs, since a stochastic producer makes single runs uninformative, consistent with this repository's own guidance. Report tokens per used memory, not per call.
- Small-corpus power will be weak. Prefer paired per-case comparisons and pre-registered deletion thresholds, and expect "inconclusive" for many pipelines. An inconclusive result should default to the cheaper, smaller-attack-surface configuration.
- A pipeline should also be judged by its marginal effect given all others, so ablate one at a time and then test the minimal set together.

### Gaps
- No public benchmark measures the overlapping-pipeline scenario or personal-assistant procedural learning.
- No independent adjudication exists for the vendor benchmark disputes.
- MemoryArena and the "Are We Ready" evaluation ([arXiv](https://arxiv.org/html/2606.24775)) were seen only through summaries or search results.
