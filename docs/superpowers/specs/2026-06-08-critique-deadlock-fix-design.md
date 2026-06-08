# Pre-execution critique deadlock fix

**Date:** 2026-06-08
**Status:** Approved (design); revised after code-grounded review (see Link 3 scope)
**Branch:** feature/sliding-window-phase0-observability — _note: this is the
sliding-window observability branch; confirm the critique-deadlock fix lands on its own
branch/PR rather than riding along with unrelated work._

## Problem

A user reported an agent stuck in a loop, unable to run `relevo --help` (a benign
read-only CLI help command). The bot told the user the command was "blocked as
underspecified" / "critique error", which read as though the external `relevo` tool
was rejecting input.

Telemetry (`db_probe --search relevo`) shows the real blocker was aidaemon's own
**pre-execution groundedness critique**, not `relevo`:

```
tool(terminal): [SYSTEM] Critique pass blocked this risky action. Issues: missing
  evidence: the user's request implies a previous 'help' command was already run,
  but there is no context or evidence...
tool(terminal): [SYSTEM] Critique pass blocked this risky action. Issues: missing
  evidence: The user's request 'Run again' is underspecified and lacks context...
```

The agent never executed `relevo --help`. Its own guardrail refused, repeatedly, and
the user was funneled into a clarifying-question dead end.

## Root cause — a closed deadlock of four independent links

Breaking any single link ends the loop, but each is a latent bug, so all four are fixed.

1. **Link 1 — the critique fires on every terminal command.**
   `Terminal::capabilities()` (`src/tools/terminal.rs:2099`) is statically
   `external_side_effect: true, high_impact_write: true`.
   `should_run_pre_execution_critique` (`src/agent/loop/tool_prelude_phase.rs:525`)
   triggers on either flag, so even a `--help` invocation gets the full critique.
   The real per-command risk (`command_risk::assess`, which scores `relevo --help`
   as `Safe`) never feeds the gate. Note also that `policy_bundle.risk_score`
   = `text_risk*0.7 + cap_risk*0.3` (`src/agent/policy/policy_signals.rs:323`) is
   driven by the user's *message text* and static capability flags — not the actual
   command — so it cannot rescue this case either.

2. **Link 2 — the critique judges the conversation, not the action.**
   `request_pre_execution_critique` (`tool_prelude_phase.rs:619`) feeds the user's
   chat message in as "User request." For vague turns ("Run again") it flagged
   `missing evidence` — reasoning that the user *implied* a prior run with no
   evidence. That is a logical trap: `--help` is the canonical way to *gather*
   evidence, and the gate blocked the command that would satisfy its own complaint.
   It conflated "missing context in the user's narration" with "missing prerequisite
   for the action." This generalizes beyond terminal: a vague message + any concrete
   action can be blocked the same way.

3. **Link 3 — no escape from repeated identical blocks.**
   The reject branch (`tool_prelude_phase.rs:1183`) re-injects the same retry message
   each time, with no degradation when the *same* tool+target is rejected repeatedly.

4. **Link 4 — the confirmation shortcut traps "yes".**
   `maybe_handle_non_resolving_confirmation_shortcut` (`src/agent/loop/bootstrap/shortcuts.rs:354`)
   intercepts a bare confirmation when the prior assistant message contained " or "
   (`question_requires_specific_answer`). The user's "Yes" was short-circuited with a
   canned "answer with the exact choice" reply *before any LLM call* — re-issued every
   turn with no progress.

## Fix

### Link 1 — gate consults real command risk
When the tool call is `terminal`, run `command_risk::classify_command()` on the
command and **skip the critique when the assessment is `RiskLevel::Safe`**.
Non-terminal tools keep current behavior. Reuses existing, well-tested risk infra;
generalizes to `ls`, `cat`, `which`, `--version`, `--help`, etc.

Implementation detail: `should_run_pre_execution_critique` currently takes only
`(policy_bundle, capabilities, execution_state)` and has no access to the command
string, so it cannot call `classify_command` as-is. Thread the `&ToolCall` (or the
parsed command) into the function, or branch in the caller before invoking it — pin
this down rather than deferring, since the Link 1 unit test presupposes the gate can
see the command.

Terminal-specific parsing:
- Extract the `command` field from the terminal tool's JSON arguments; the `run`
  action is the one that executes a shell command and should be risk-gated. `check`
  / `kill` (PID-based) carry no shell command — fail safe (run the critique, or treat
  as not-Safe) when no command string is present or args are malformed.
- `classify_command` already splits on pipes/operators and returns the max segment
  risk, so a piped command with any non-Safe segment stays gated. Good.
- `hard_block_reason()` and terminal's interactive approval (`needs_approval`) apply
  independently of this skip — skipping the critique does not weaken either.
- Note for the record: an unknown binary with `--help` (e.g. `relevo --help`) falls
  through to `Safe` today — `command_risk.rs` has no explicit `--help` handling; it is
  Safe because nothing flags it. That is the intended behavior here, not an accident.

Correction: the public API is `command_risk::classify_command`, not `assess()`.
`RiskLevel` variants are `Safe | Medium | High | Critical`.

### Link 2 — critique judges the action, not the conversation
Reframe the critique system + user prompt (`request_pre_execution_critique:619`) to
evaluate only the proposed action's **safety, target correctness, and verifiability**.
Add an explicit instruction: *a short or vague user message is not, by itself,
"missing evidence" — only flag missing evidence when the action genuinely depends on
a prerequisite that has not been established.* Keep `user_text` in the prompt for
"wrong target" detection.

No deterministic guard is added for Link 2. Links 1 and 3 already provide the
deterministic backstops (Link 1 prevents the misfire for safe terminal commands;
Link 3 is a tool-agnostic loop-breaker). A Link-2-specific guard keyed on
`command_risk::Safe` would be redundant with Link 1, and a more general guard would
require fragile string-matching on the critique's free-text issue that risks masking
legitimate groundedness blocks (e.g. editing a file that was never read).

**Related guard, not in scope.** A separate uncertainty-clarify guard
(`tool_prelude_phase.rs:~850`, gated by `policy_config.uncertainty_clarify_enforce`
+ `uncertainty_score >= threshold`) can *also* end a turn with a clarifying question
on a vague message + side-effecting tool, independent of the critique. It is not the
reported failure mode, but it produces a similar "vague message → dead end" UX.
Implementers should not assume Link 2 resolves all vague-message stalls; this guard is
a distinct path and is out of scope here.

**Residual cost (known and accepted).** Link 1 only covers terminal. For *non-terminal*
tools, the vague-message→missing-evidence conflation is mitigated only by the Link 2
prompt reframe (weak coverage) plus Links 4 — Link 3's ephemeral counter does **not**
catch the cross-turn case (see Link 3). This residual cross-turn loop is currently
**unobserved in telemetry**; it is consciously left to Links 1/2/4 + instrumentation
rather than absorbed by new persisted state. If telemetry later shows it, the
event-log-derived upgrade path in Link 3 closes it without schema changes.

### Link 3 — escape hatch after repeated identical blocks
Track consecutive critique rejections keyed by `(tool_name, target_hint)`. After
**2 identical rejections**, the 3rd attempt bypasses the critique and the agent
surfaces the truth to the user instead of silently retrying. Then it proceeds. The
counter resets when a different tool/target is attempted or the action succeeds.
N=2 mirrors the existing `verification_block_count >= 2` escape pattern in
`completion_phase.rs` (one difference: that escape clears its guard silently; Link 3
is deliberately user-visible — keep that distinction intentional).

**Counter scope — the critique fires at most once per turn, so the loop is cross-turn.**
`set_plan_version` (`tool_prelude_phase.rs:1007`) runs before the critique call, so on a
rejection `current_plan_version` is already set; the next iteration in the same turn
hits the `plan_already_generated` gate (`tool_prelude_phase.rs:975`) and skips the
entire plan+critique block. The reported deadlock is therefore **cross-turn**: the
model reacts to the injected "[SYSTEM] Critique pass blocked… re-plan / gather
evidence" message by asking the user to clarify, the user replies ("Run again", "Yes"),
and each reply starts a *fresh* `ExecutionState` (built per-turn in `main_loop.rs`)
whose critique runs and rejects again. A counter living in per-turn `ExecutionState`
resets every turn and never reaches the escape threshold — a per-turn unit test would
pass while the real loop persists.

**Decision: ephemeral intra-turn counter now; derive cross-turn count from the event
log only if telemetry demands it. Do NOT add a persisted counter table.** Rationale —
long-term cost discipline for a self-hosted open-source agent:

- The *reported* incident is terminal + `Safe` and is fully resolved by **Link 1**
  alone. There is currently **no telemetry** of the non-terminal cross-turn loop.
  Building durable persisted state for an unobserved loop is speculative — exactly the
  kind of permanent migration/cleanup/contributor-cognition liability that rots an
  open-source codebase. A safety guard should fail toward *ephemeral* leniency and
  "surface to the human," never toward a *durably persisted* auto-bypass of its own
  check (especially for non-terminal tools, which have no `needs_approval` backstop).

- So **Link 3 ships as an ephemeral in-`ExecutionState` counter**, honestly documented
  as defense-in-depth that rarely fires given the once-per-turn gate. Cross-turn
  coverage is owned by **Links 1 + 2 + 4** plus the existing rejection telemetry
  (`DecisionType::ExecutionCritiquePass`, `tool_prelude_phase.rs:1161`).

- **Upgrade path if telemetry shows the cross-turn non-terminal loop in the wild:**
  derive the count from already-persisted, immutable state — **scan recent
  `ExecutionCritiquePass`-rejected events for the session matching `(tool,
  target_hint)`** (the decision-point payload at `:1170` already records
  `tool` + `target_hint` + `critique_result`). Zero new schema, append-only source,
  self-pruned by existing event machinery, testable by seeding events. Read this from
  the event store, **not** the in-context message view — the sliding-window collapse
  strips old tool intermediates, so a context scan would miss prior blocks. Reach for a
  bespoke persisted counter only if the event-log derivation proves insufficient.

**Bypass message must be conditional — do not assert safety blindly.** The proposed
copy *"…it looks safe, so I'm running it now"* is only true when the action was
actually risk-assessed Safe — i.e. a terminal command that `classify_command` returned
`Safe` for. Link 3 itself consults only a repetition counter, not risk, and applies to
*all* tools. For a non-terminal action — e.g. `edit_file` on a file that was never
read, which is exactly the *legitimate* groundedness block the Link 2 rationale says
must be preserved — Link 3 would both override a valid block and tell the user it
"looks safe" when nothing checked. Resolve the tension:
- Terminal + `classify_command == Safe`: keep the confident copy and run.
  > "My groundedness check has refused `<command>` twice; it looks safe, so I'm running it now."
- Otherwise (non-terminal, or non-Safe command): do **not** claim safety. Surface that
  the check keeps blocking and either proceed with a neutral message or stop and report,
  rather than auto-running a possibly-legitimate block under a false "looks safe" banner.
  > "My groundedness check has refused `<tool/target>` repeatedly; I'll surface that rather than keep retrying."

Terminal's separate interactive approval flow (`needs_approval`) still independently
gates genuinely dangerous *terminal* commands, so this cannot auto-run a destructive
shell command. Note there is no equivalent approval backstop for non-terminal tools —
another reason the non-terminal bypass should not silently run under a safety claim.

### Link 4 — anti-loop pass-through in the confirmation shortcut
Before re-issuing the canned nudge in
`maybe_handle_non_resolving_confirmation_shortcut`, check whether the *same* nudge was
the immediately preceding assistant turn. If it was, let the bare confirmation fall
through to the LLM instead of re-nudging. Breaks the loop after exactly one nudge;
preserves the helpful first nudge for genuine "X or Y" questions.

Detection must use a **stable matcher, not exact-text comparison.**
`build_specific_answer_request` (`shortcuts.rs:656`) emits one of *two* templates
(short-question reuse vs. generic fallback), but both begin with the literal
`"I still need the specific"`. Match on that stable prefix (or add a dedicated marker
to the nudge) so both templates trigger pass-through; matching full prior text is
brittle.

Also: the pass-through must cover **all** bare confirmations, not just "yes".
`is_bare_confirmation` already recognizes "sure", "go ahead", "ok", etc., so the loop
can form with any of them — the test plan must exercise more than the literal "yes".

### Honesty note (related, lower priority)
The agent paraphrased its own internal critique block as if `relevo` had rejected the
input. Not addressed in this spec, but worth a follow-up: internal critique/guard
reasoning should not be presented to the user as an external tool's error.

## Sequencing

All four are real bugs and all four ship, but the leverage is uneven:

1. **Link 1** — highest leverage; likely resolves the reported `relevo --help` incident
   on its own. Land first.
2. **Link 4** — cheap, no LLM cost; fixes the "Yes"/"sure" trap. Land second.
3. **Link 2** — prompt-only change; low cost, uncertain benefit.
4. **Link 3** — ship as an **ephemeral intra-turn counter** (defense-in-depth behind
   Link 1); do not add persisted state. Cross-turn coverage stays with Links 1/2/4 +
   telemetry; the event-log-derived count is the documented upgrade path if data shows
   the cross-turn loop occurring.

## Testing

Project gate: `cargo fmt`, `cargo clippy --all-features -- -D warnings`, `cargo test`.

- **Link 1:** unit test — the terminal-aware gate returns `false` for `relevo --help`
  (Safe) and `true` for a Critical command (e.g. `rm -rf /`). Edge cases:
  `action: "check"` / malformed args / missing `command` → critique still runs (fails
  safe); a piped command with one non-Safe segment → critique still runs.
- **Link 3:** unit test on the ephemeral intra-turn counter — escapes on the 3rd
  identical `(tool, target)` rejection within a turn; resets on a different target or on
  success. Assert the bypass does **not** skip the evidence gate or terminal approval,
  and that the bypass message asserts "looks safe" only on the terminal-Safe path
  (non-terminal / non-Safe → neutral message). Document explicitly that the counter is
  per-turn and does **not** cover the cross-turn loop (owned by Links 1/2/4). If the
  event-log-derived cross-turn upgrade is later implemented, add a test that seeds N
  `ExecutionCritiquePass`-rejected events for `(tool, target_hint)` and asserts escape.
- **Link 4:** unit test — a bare confirmation passes through when the prior assistant
  turn matched the `"I still need the specific"` prefix; still intercepted on first
  occurrence. Exercise **both** nudge templates and more than one confirmation token
  ("yes", "sure", "go ahead").
- **Link 2:** prompt-text assertion that the new scoping instruction is present, plus
  an integration test (`MockProvider`) that a vague message + safe command is not
  blocked. Script the mock critique LLM to return `replan` + `missing evidence` on
  vague text and assert Link 1 still wins for a Safe terminal command. This link has
  inherently weaker automated coverage (LLM judgment); called out explicitly.
- **Composite acceptance (end-to-end):** the scenario that validates the four links
  together, not just unit slices —
  1. vague user message ("Run again"),
  2. agent proposes `terminal` with `relevo --help`,
  3. mock critique would reject,
  4. the command still executes (Link 1) — or, on the "Yes" path, the bare confirmation
     reaches the LLM instead of looping (Link 4).

## Out of scope
- Reworking `policy_bundle.risk_score` weighting.
- The honesty/sanitization follow-up above.
- Any change to terminal's interactive approval flow.
