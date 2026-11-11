# Decision layer architecture

Status: implemented 2026-11-11 (crate `aidaemon-decisions`, daemon
`src/llm_runtime/decisions.rs` + `decide.rs`), off by default. See
"Implementation notes" at the end for where the build differs from this
design.
Date: 2026-11-11.

## In a nutshell

A small, optional layer that answers **closed questions** (yes or no, pick
one of N, a score) quickly and with an honest confidence, through one gate
in the daemon. It makes the agent faster and its choices better without
touching the main model's work:

1. **Turn digest**: one ~0.3 s call per incoming message replaces the
   relationship-router chat call (~1.9 s today, part of it a failover that
   should be fixed anyway; ~0.8 s for a healthy chat call) and the English
   keyword checks (identity pinning, resume). It does not depend on English
   wording; other languages are not yet tested.
2. **Recall filter**: drops remembered facts that are irrelevant to the
   message before they reach the prompt (76% of injected facts were
   irrelevant on real traffic).
3. **Model routing** (later phase): sends a request to the fast, primary
   or smart model by how demanding it is, choosing only when a new request
   starts.

When the layer is off, unsure or failing, the daemon does exactly what it
does today. A decision never grants authority, decides completion or
loosens a safety check.

The speed gain, fact dropping, routing down, resume and the command check
need a backend with calibrated confidence, and today only Gemini's decision
model is one. On a chat-model backend the layer gives identity pinning that
does not depend on English wording, a relationship answer no better or
faster than today's router, and shadow data.

## Why

aidaemon is meant to be one agent for everything a person would otherwise do
in the ChatGPT or Claude apps, and it is open source, so it runs for many
people, in many languages, on providers we do not choose. The owner's goals
for this layer, in order: speed, correctness (better decisions), and lower
token use.

Several of the daemon's small decisions are closed questions. Today they are
answered either by a full chat-model call that writes JSON (slow, parsed,
failure-prone) or by English keyword lists (fast, brittle). A decision layer
answers them through one typed interface.

## Evidence

All numbers come from spikes run on 2026-11-11: synthetic cases plus replays
of the owner's last 14 days of real traffic through Google's
`gemini-decision-1` and, for comparison, `gemini-3.5-flash-lite` asked for
JSON. Labels where needed were judged by the model that ran the spikes, in
the same session that produced this design, and not blind to which backend
gave which answer. Conversation context was rebuilt from the event log (no
session summary, no owed-request rows), so the replays saw less than the
live router did and less than the design below gives the digest. Much of
the traffic is live testing, and no language other than English was tested
on its own. Treat numbers
as directional; none of them replaces shadow data.
The raw data contains real messages and stays out of the repository.

| Question | Finding |
|---|---|
| Speed | `gemini-decision-1`: p50 ~280 ms, p95 ~420 ms at ~1.3k input tokens with 1 to 12 questions; ~3,000 calls, 0 API errors. Chat model: ~830 ms. |
| Time before the main model starts | p50 3.0 s, p90 9 s since 2026-11-09. The relationship router takes ~1.9 s on the half of turns where it runs; memory recall rendering is the other large part. The 1.9 s includes the primary model rejecting the call and the failover answering (see "Related work"), so it overstates what a healthy router costs. |
| Relationship router | 66% agreement with the logged router. On 25 labelled disagreements: decision model 18 right, chat model 18, old router 7. Only disagreements were labelled, so this says nothing about the turns where both agreed, and the replay had less context than the router. |
| Confidence honesty | Over 270 labelled router cases, answers at confidence ≥0.9 that were wrong: decision model 1, chat model 35 (self-reported) or 32 (agreement over 3 samples). Current Gemini chat models refuse log probabilities. Accuracy between 0.6 and 0.9 was not measured. |
| Confidence on real traffic | Decision model: 71% of router answers below 0.9, 15% below 0.6. Thresholds must come from data, not from synthetic tests. |
| Identity pinning | Keyword list pins "Say: my name is fluffy" and misses "my daughter's names?", "my spouse's name". The decision model caught 38 identity messages the list missed. |
| Recall relevance | 1,440 injected facts over 200 turns: 24% judged relevant; 43% of turns had none relevant; near-duplicate facts common. Keeping identity facts always and dropping others only below P(relevant) 0.15: 0 of 8 facts the real reply used were dropped, 47% of unused facts removed. The chat model at the same setting dropped 3 of 8 used facts. Eight used facts is too few to call the rule safe (0 of 8 is consistent with a true drop rate up to about 30%), and whether fewer irrelevant facts improve replies was not measured. |
| Model tier | Tiers track real work: median tools used fast 1, primary 3, smart 8. Of 104 fast-tagged turns, 21 did real work (3 or more tools); 4 of those 21 had confidence ≥0.8 and would still have been routed down. How many of the 104 were at ≥0.8 was not recorded, so the error rate of routing down at that threshold is unknown. (Chat model: 13 of 13.) |
| Prompt cache | Recalled facts are placed after the user message, so changing them does not break the cached prefix (cache reuse 76% gpt-6-sol, 94% Haiku on later calls). Irrelevant facts cost fresh tokens and attention, not cache. |
| Tool selection | Not worth it: even offering 4.6 of 10 tool groups, a needed group was missing in 4% of turns. Tool definitions (~9k tokens) sit in the cached prefix. |
| Memory extraction gate | Not worth it: a pre-check sees the message, not the work, and would have skipped 30 of 50 turns that produced useful work memories. |
| Command second opinion | Flags 5% of commands at 0.9, including unasked deploys and download-and-run commands, with some noise. Unknown how many were already covered by owner grants. |

## Decided by the owner

- The layer is optional. Without it, the daemon behaves exactly as today.
- It is additive: it does not replace the main models or remove existing
  side calls; it decides whether to do work or which existing path to take.
- It is provider-agnostic, including local models. Gemini's decision model
  is one backend, not a dependency.
- There is one central, callable gate.
- It fits the SDK separation on the PR #14 branch: a reusable crate behind a
  public trait, with the daemon wiring it in.
- Memory exists to make the agent better at the work done together (what was
  built, how, what worked and failed), not only personal facts.

## Goals

1. A provider-neutral `Decider` trait in the SDK, with a chat-model backend
   that works on any `aidaemon_core::Provider` and a Gemini Decisions API
   backend.
2. One daemon entry point, `SideCallGateway::decide`, through which every
   decision runs, is recorded and falls back to its default.
3. Every answer carries where its confidence came from, and every decision
   declares which confidence sources it may act on.
4. Shadow mode, so each decision is measured on live traffic before it acts.
5. The three decisions in "Decisions" below.

## Non-goals

- Replacing the main agent model's work, or anything that writes text.
- Choosing which tools to offer (see Evidence; revisit as model-driven tool
  search if tool counts grow large, not as a pre-turn classifier).
- Gating memory extraction (see Evidence).
- Running tools.

## Anti-goals (failure even if it works)

- **A decision decides authority, completion, trust or approval.** These
  stay with explicit `Authority`, receipts and kernel hooks (CLAUDE.md:
  "Evidence beats contract", "Owner Approval Rule", "Untrusted Content
  Rule"). A decision may make a safety check stricter, never looser.
- **A confidence-gated decision acts on an uncalibrated backend.** The chat
  model is confidently wrong too often (Evidence); a rule like "drop the
  fact only when sure" is unsafe on it.
- **The layer becomes a new phrase classifier.** Where typed state answers a
  question (taint, session, open checkpoint, fact category, mutation
  effects), typed state answers it. A decision is for questions where the
  language itself is the input.
- **A required dependency.** No build, test or default config needs a Google
  key or network access.
- **Silent behavior change.** A decision acts only after its shadow data has
  been reviewed.
- **Slower turns.** A decision on the turn path runs in parallel with work
  already there, or replaces a call that was already there.

## The Gemini Decisions API (reference)

Source: Google AI Studio docs ("Decisions API", pricing), read 2026-11-11,
and the spikes above. Re-check before building `GeminiDecider`; the API is
`v1beta`.

- `POST https://generativelanguage.googleapis.com/v1beta/decisions`, header
  `x-goog-api-key`. Model `gemini-decision-1`.
- Request: `{ "model", "input", "questions": { "<key>": { "type": "noul" |
  "choice" | "score", "instructions", "criteria" } } }`. `input` is a string,
  one content block or an array (`text`, `image`, `audio`, `video`,
  `document`). Tool-call, tool-result and thought blocks are refused.
- `noul` (yes/no): optional `criteria` with `true_description` and
  `false_description`. Returns `{ "noul": p_yes, "confidence" }`; `p_yes` lies
  in 0.01..0.99 and `confidence = max(p, 1 - p)`.
- `choice`: `criteria` maps option ids to descriptions, 2..255 options (a
  single option is a 400), built per request. Returns `choice`,
  `probabilities` (sum to 1 even when nothing fits; include a catch-all) and
  `confidence`.
- `score`: ordered level descriptions, 2..255. Returns `score` (weighted mean
  of level indices from 0), `legend`, `probabilities`, `confidence`.
- Response also carries `usage` (`total_input_tokens`; output is always 0).
- Limits: 1..1,000 questions within a 262,144-token context; question keys
  `[A-Za-z0-9_-]{1,64}`; option keys up to 128 characters. Questions run in
  one parallel pass and cannot see each other's answers.
- Errors: HTTP 400 for limits, bad keys and safety blocks (`error.code`
  `safety`, `prohibited_content`, `blocklist`, `image_safety`,
  `content_blocked`).
- Billing: input tokens only, $0.10 per million (standard tier); free tier
  exists. No Batch API, no context caching.

## Shape

```text
aidaemon-core (Provider, Message, ...)
   ^
aidaemon-decisions            <- new SDK crate
   Decider trait, question/answer types, ConfidenceSource,
   ProviderDecider (any Provider), GeminiDecider (feature "gemini"),
   ScriptedDecider (feature "testing"), conformance kit
   ^
daemon: src/llm_runtime/
   decisions.rs   registry: enum Decision + DecisionSpec
   gateway.rs     SideCallGateway::decide (the one door)
   door_tests.rs  no Decider used outside the gateway
   ^
call sites: turn bootstrap (digest), daemon recall render (filter),
            turn model selection (routing)
```

### Why a separate crate

A decision backend is not a `Provider`: it generates no text and returns
probabilities per question. Forcing it through `Provider::complete` would
hide the probabilities and invite parsing. A separate crate keeps
`aidaemon-providers` about chat models and adds one row to
`crates/EXTENDING.md`:

| To add | Implement | Crate | Template to copy |
|---|---|---|---|
| A decision backend | `Decider` | `aidaemon-decisions` | `ProviderDecider` |

The crate depends only on `aidaemon-core`, `serde`, `async-trait`,
`thiserror`, and, behind `gemini`, `reqwest`. It knows nothing about the
daemon, side calls or config.

### Types (crate)

```rust
pub struct DecisionRequest {
    /// Shared context every question is asked about.
    pub input: Vec<InputBlock>,
    /// Keyed questions; keys match `[A-Za-z0-9_-]{1,64}`.
    pub questions: BTreeMap<QuestionKey, Question>,
}

pub struct Question { pub instructions: String, pub kind: QuestionKind }

pub enum QuestionKind {
    YesNo { true_description: Option<String>, false_description: Option<String> },
    /// Option key -> description, 2..=255 options, built per request.
    Choice { options: BTreeMap<OptionKey, String> },
    /// Ordered level descriptions, lowest first, 2..=255.
    Score { levels: Vec<String> },
}

pub struct Answer {
    pub value: AnswerValue,          // YesNo(p_yes) | Choice(OptionKey) | Score(f32)
    pub probabilities: Option<BTreeMap<String, f32>>,
    pub confidence: f32,             // probability the top answer is right
    pub confidence_source: ConfidenceSource,
}

pub enum ConfidenceSource {
    /// Calibrated probabilities from a decision model.
    Calibrated,
    /// Derived from token log probabilities of a constrained answer.
    TokenLogprob,
    /// The model stated its own confidence; measured as unreliable.
    SelfReported,
}

pub struct DecisionResponse {
    pub answers: BTreeMap<QuestionKey, Answer>,
    pub usage: aidaemon_core::Usage,
    pub backend: String,
}
```

Every type is `#[non_exhaustive]` with constructors, so fields can be added
without breaking third-party backends.

### Trait (crate)

```rust
#[async_trait]
pub trait Decider: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> DeciderCapabilities; // max_questions, media, confidence_source
    async fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse, DecisionError>;
}
```

`DecisionError` mirrors `ProviderError` where it can (rate limited,
transient, auth, invalid request, safety block). A backend answers every
question or returns an error; never a partial set.

### Backends (crate)

- **`ProviderDecider`** wraps any `Arc<dyn Provider>` and a model name, asks
  all questions in one request with `ResponseFormat::JsonSchema` (one
  property per question, enum-constrained for choices and levels, plus a
  confidence field) and validates the reply. Confidence is `SelfReported`.
  This is the default and the local-model path. It does not sample several
  times: agreement over 3 samples did not make the confidence trustworthy
  and tripled cost.
- **`GeminiDecider`** (feature `gemini`) calls the Decisions API.
  Confidence is `Calibrated`. Whether it shares HTTP transport code with
  `aidaemon-providers` is left to the builder.
- **`ScriptedDecider`** (feature `testing`) returns fixed answers.
- **Not now: `LogprobDecider`.** Core exposes no log probabilities, and the
  current Gemini chat models refuse them. The `TokenLogprob` variant exists
  so a provider that offers them can be added without a breaking change.

A conformance kit checks that a backend answers every question with the
declared kind, keeps probabilities in range (summing to 1 for choices),
reports its confidence source and maps errors correctly.

### Registry and gate (daemon)

`src/llm_runtime/decisions.rs` mirrors `side_calls.rs`:

```rust
pub enum Decision { TurnDigest, RecallFilter, ModelRoute, CommandRiskSecond }

pub struct DecisionSpec {
    pub purpose: &'static str,         // telemetry key, unique
    pub effect: DecisionEffect,        // what acting on it may do
    pub deadline: Duration,
    /// Per question key: minimum confidence and the sources it may act on.
    pub acts_on: &'static [(QuestionKey, f32, &'static [ConfidenceSource])],
}

pub enum DecisionEffect {
    /// May skip or remove optional work or context. Default: keep it.
    SkipOptional,
    /// May pick one of several paths that hold the same authority. A wrong
    /// pick can still cost quality, so each such decision states that cost.
    Route,
    /// May only make a check stricter.
    EscalateOnly,
}
```

Call sites build the questions (they need the turn's data); the spec holds
everything that governs whether an answer is acted on.

`SideCallGateway::decide(decision, request) -> DecisionOutcome`:

```rust
pub enum DecisionOutcome {
    Decided(GatedResponse),
    UseDefault(DefaultReason), // Disabled | Shadow | Failed | Timeout
}
// GatedResponse is a daemon type: the crate's DecisionResponse plus the
// spec's `acts_on`. GatedResponse::acted(key) -> Option<&Answer>, None when
// the answer is below its threshold or from a source the spec does not trust.
```

The gate:

1. Returns `UseDefault(Disabled)` at once when no backend is configured.
2. Runs the configured `Decider` under the spec's deadline.
3. Records one telemetry event per call (purpose, backend, latency,
   per-question value, confidence and source, whether each answer was acted
   on, usage) with `call_purpose = "decision:<purpose>"`.
4. In shadow mode, records and returns `UseDefault(Shadow)`.
5. Applies `acts_on` per question, so a digest can use the router answer
   while the resume answer falls back.

`door_tests.rs` gains a check that no code outside the gateway holds a
`Decider`.

### Configuration (daemon)

```toml
[decisions]
# "off" (default) | "provider" | "gemini"
backend = "off"
# For backend = "provider": which configured model to ask (default: cheap).
model = "cheap"
# Per decision: "off" | "shadow" | "on". Everything starts in shadow.
[decisions.mode]
turn_digest = "shadow"
recall_filter = "shadow"
model_route = "off"
command_risk_second = "off"
```

The Gemini key is read from the keychain like other provider keys.

## Decisions

### 1. Turn digest (`Route`, turn path)

One call per incoming owner message, **started when the message arrives and
awaited where its answers are used**. At ~0.3 s against a median 3.0 s of
existing pre-turn work, awaiting it adds nothing noticeable, and it replaces
the router's chat call. Questions:

- **relationship**: one choice that carries the antecedent, built per turn:
  `new_request`, `clarification_answer`, `courtesy`, and one
  `continuation:<message id>` option per candidate earlier user message. The
  classes are the router's existing ones
  (`src/agent/turn/bootstrap/task_planning.rs`), no new class. It is one
  question, not two, because questions cannot see each other's answers and
  the router's contract ties them together (a continuation names exactly one
  antecedent, every other class names none). Same context the router gets
  today (recent messages with ids, session summary, owed request rows).
  A wrong answer does not change authority but does change behavior: a
  wrong `new_request` detaches the turn from the request it continues, and
  a wrong antecedent binds it to the wrong one.
- On turns where the router is bypassed today (the bypass rule in
  `task_planning.rs`), the relationship answer is recorded and not acted
  on. Whether the digest should act there is a question for shadow data.
- **resume**: whether the message asks to resume the open checkpoint. Only
  asked when typed state shows an open checkpoint for this session and
  owner. Replaces `is_resume_request` (`src/agent/agent_helpers.rs:459`).
- **identity**: whether the message states or asks about the identity of the
  owner, the assistant or family. Stored as a typed flag on the message row;
  the renderer (`src/agent/turn/turn_render.rs:491`) pins a message if the
  flag **or** `text_relates_to_critical_identity` says so, so older
  messages and messages from when the layer was off keep today's behavior.
  The union fixes the keyword list's misses, not its false pins ("Say: my
  name is fluffy" is still pinned); removing those means letting the flag
  override the keyword check, a later step once shadow data supports it.
- **tier** (shadow only until decision 3 is built): which model tier the
  message needs, recorded for measurement.

Acting thresholds: relationship acts at confidence ≥0.6 on `Calibrated`.
That number is a starting guess: honesty was measured only at ≥0.9, and
more than half of real answers fall between 0.6 and 0.9, so shadow data
must show accuracy in that band before the decision acts. On `SelfReported`
no threshold means anything (the chat model reports ≥0.9 when wrong), so
the answer is acted on as given, which matches today's router only if the
configured model is at least as strong as the one the router uses; with a
weaker `model`, relationship stays in shadow. Identity acts at p ≥0.5 on
any source (the union rule makes extra pins cheap). Resume acts at ≥0.8 on
`Calibrated` only; otherwise the keyword check decides.

Fallback: below threshold, timeout or error, the relationship falls back to
today's router call (serial, slower on those turns, never less correct).
The deadline bounds that cost: at p95 ~0.4 s, a deadline near 0.6 s keeps
the worst case close to today's, where 1.5 s would add most of a second
router call. With the layer off, the router runs as today and the keyword
checks stay.

### 2. Recall filter (`SkipOptional`, turn path)

After memory recall selects facts and before they are rendered into the
prompt, ask one yes/no per selected fact: "would this fact help answer or
carry out the message?", with the recent conversation as context.

- Facts in identity categories (the ones `IDENTITY_CATEGORIES` already
  treats as critical, including `user`) are **always kept**; this is typed,
  not asked.
- Other facts are dropped only when P(relevant) < 0.15 **and** the source is
  `Calibrated`. On any other source, nothing is dropped (shadow records
  only).
- It lives in the daemon's own memory system (`DaemonRecall`,
  `src/agent/memory_plugin/recall.rs`), so the host names no memory concepts
  outside it. Another memory system may take a `Decider` itself.
- It must not add serial time: it runs after recall's lookups, so the
  builder starts it as soon as the fact list exists and overlaps it with
  the remaining bootstrap work. If it cannot be overlapped, accepting
  ~0.3 s is an exception to the "Slower turns" anti-goal for the owner to
  grant, not a builder choice.
- Before it acts: deduplicate facts at write time (see "Related work") and
  re-measure, since duplicates are part of the 76% and are cheaper to
  remove at the source; then gather enough shadow turns to bound the rate
  of dropping a fact the reply used (the replay had 8 such facts), and
  check whether replies improve, since the filter saves no cache and only
  a few hundred fresh tokens a turn.

### 3. Model routing (`Route`, later phase)

After the digest has run in shadow and its `tier` answers have been compared
with the work turns actually did:

- Choose the tier only when a turn starts a new request; a continuation
  stays on the model its task started on (keeps the prompt cache and
  continuity).
- Route **down** to the fast model only at high confidence (start at ≥0.8)
  on `Calibrated`; route **up** to the smart model at ≥0.6. Anything else
  stays on primary. A chat-model backend never routes down.
- A turn on the fast model can hand off upward on typed signals (repeated
  tool failures, an open checklist beyond a size), never by reading its own
  text.
- Uses the existing `fast` / `primary` / `smart` tiers (`src/router.rs`).

### 4. Command second opinion (`EscalateOnly`, optional)

One yes/no per effect in `APPROVAL_EFFECTS`
(`src/tools/semantic_command_risk.rs`), asked next to the primary
assessment, only where that assessment runs today (the last branch of the
terminal approval ladder in `src/tools/terminal.rs`; never on allowlisted,
granted or trusted-schedule commands). A flag the primary missed requires
approval; it never removes one. `Calibrated` only, threshold ≥0.9. Off by
default: real traffic suggests about two extra prompts a day, some useful
(unasked deploys, download-and-run), some noise.

## Not building, with reasons

- **Tool selection.** Misses a needed tool group in 4% of turns even when
  generous, saves tokens that are already cached, and per-turn changes
  break the cache.
- **Memory extraction gate.** Work knowledge lives in what the agent did,
  not in the message; the gate would have skipped 60% of useful work
  memories.

## Related work outside this layer

Found during the investigation; separate changes:

- Profile memory recall (`DaemonRecall::render_block` awaits its lookups one
  after another); it is the other large part of the 3 s before the main
  model starts.
- Deduplicate facts at write time (the same module path stored several
  times; five near-identical "disable deep reasoning for GUI" facts) and
  treat volatile facts ("latest commit", temporary URLs) as such.
- Delete the `#[cfg(test)]`-only keyword helpers in
  `src/agent/intent/intent_routing.rs`, `src/agent/runtime/followup.rs` and
  `src/agent/intent/relational_prefilter.rs`.
- The ChatGPT subscription models reject the router and verifier side calls
  instantly (158/158, 276/882 in a week), so Haiku answers through
  failover. Worth diagnosing regardless of this layer.

## Untrusted content and privacy

- Decision input may contain text someone other than the owner wrote.
  Because no decision grants authority or loosens a check, injected text can
  at worst misroute between paths that hold the same authority (including
  detaching a turn from its request or binding it to the wrong one), keep
  or drop a fact, or pick a model tier. `EscalateOnly` decisions can only add prompts.
- Sending decision input to a hosted backend is egress of owner data, like
  any provider call. The backend is the user's explicit choice and off by
  default. Google's free tier may use submitted data to improve its
  products; the `gemini` backend docs must say so.
- The layer reads no secrets and runs no tools.

## Measuring

- `db_probe --decisions --eval-hours N`: per decision and backend, call
  count, p50/p95 latency, deadline hits, outcome mix, acted-on rate per
  question, and in shadow mode agreement with the path actually taken.
- Turn digest: time from message to main-model start (before/after, with
  the baseline re-taken once the router's failover is fixed), agreement
  with the router on labelled cases including a sample where both agree,
  and accuracy by confidence band (below 0.6, 0.6 to 0.9, above 0.9).
- Recall filter: facts dropped per turn, and drops of facts the reply then
  used (a reply that names a dropped fact's value is a regression signal).
- Model routing: shadow tier against tools and model calls the turn used;
  N-of-M live runs before acting.
- Labelled synthetic cases (English and at least one other language) for
  each decision live in the repository; real messages never do.

## Rollout

1. Crate: types, trait, `ProviderDecider`, `ScriptedDecider`, conformance
   kit.
2. Daemon: registry, gate, config, telemetry, door test. All decisions off.
3. `GeminiDecider` behind its feature.
4. Turn digest in shadow, then on (relationship and identity first, resume
   after). Fix the router's rejected side calls first, so the speed gain is
   measured against a healthy router.
5. Recall filter in shadow, then on for calibrated backends, after
   write-time deduplication.
6. Model routing from the digest's shadow tier data.
7. Command second opinion, only if wanted.

## Left to the builder

- Exact names of types and modules, and the JSON schema `ProviderDecider`
  sends.
- How `GeminiDecider` shares HTTP transport code with `aidaemon-providers`.
- Telemetry field names, as long as `db_probe` reports the measures above.
- Default deadlines (start near 0.6 s on the turn path) and the starting
  thresholds above, tuned from shadow data.

## Implementation notes (2026-11-11)

- **Chat backend and the relationship.** A self-reported relationship answer
  is never acted on; on the `provider` backend the router keeps deciding and
  the digest's relationship is recorded only. (Replaces "acted on as given
  if the model is at least as strong", which code cannot judge.)
- **Shadow never waits.** The digest is awaited only when an answer of it is
  acted on this turn (relationship or model routing `on`); otherwise it runs
  beside the turn, and the identity flag lands on the stored message when it
  arrives. Resume in shadow also runs beside the turn. Recall filtering
  starts when the fact list exists and is awaited only where facts render.
- **Resume** is its own `Decision::Resume` call (it happens before the turn
  context exists), asked only when an active task exists.
- **Identity** is a typed `MessageAnnotation::IdentityCritical`, added to the
  stored `user_message` event after the digest answers. Finding: archived
  rendering already keeps every user message verbatim, so today the pin (and
  the keyword check it unions with) changes nothing in the prompt. It is kept
  as typed state for a consumer that needs it (critical-fact injection).
- **Gemini key.** `[decisions] backend = "gemini"` with no `api_key` uses the
  key of the first `google_genai` provider in `[provider]`, so one Gemini key
  serves model calls (as a fallback provider) and decisions.
- **Command second opinion** runs in parallel with the primary assessment
  and can only add an approval.
- **Measuring:** `db_probe --decisions --eval-hours N`. Each call writes a
  `decision_point` event (`condition = decision_layer`) and, for a backend
  that is not a side call, an `llm_call` row under `decision:<purpose>`.
