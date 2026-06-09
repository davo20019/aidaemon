# Pre-execution critique deadlock fix

**Date:** 2026-06-08
**Status:** Approved (design); revised after code-grounded review — ships Links 1/2/4,
drops Link 3 (counter-based hatch is dead code; cross-turn version deferred to telemetry)
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

Breaking any single link ends the loop, and each is a latent bug. This spec fixes
Links 1, 2, and 4. Link 3 (a generic escape hatch) is **dropped** — analysis below
shows a counter-based hatch would be dead code, and a working cross-turn version is
unjustified without telemetry; its design is documented for later.

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
command and **skip pre-execution planning *and* critique when the assessment is
`RiskLevel::Safe`**. Non-terminal tools keep current behavior. Reuses existing,
well-tested risk infra; generalizes to `ls`, `cat`, `which`, `--version`, `--help`, etc.

Implementation location — skip at the **plan-block entry**, not just the critique
sub-gate. The expensive `request_pre_execution_plan` LLM call
(`tool_prelude_phase.rs:991`) runs *before* the critique and is gated only by
`!plan_already_generated` (`:976`), **not** by `should_run_pre_execution_critique`.
Skipping only the critique would still cost one planning LLM call on every safe
terminal command — defeating Link 1's own goal of cheaply clearing `ls`/`cat`/`--help`.
Put the terminal-Safe check at the top of the plan block (`~:980`), where
`first_risky_tool_call` is already in scope (it is also used at `:1080`), so a Safe
terminal command bypasses the whole plan+critique path. The unit test should target
that gate (a small helper, e.g. `should_run_pre_execution_gating(tool_call)`, keeps it
testable in isolation).

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

No deterministic guard is added for Link 2. **Link 1** is the deterministic backstop
for the observed case (safe terminal commands). A Link-2-specific guard keyed on
`command_risk::Safe` would be redundant with Link 1, and a more general (tool-agnostic)
guard would require fragile string-matching on the critique's free-text issue that
risks masking legitimate groundedness blocks (e.g. editing a file that was never read)
— which is why Link 3 is dropped rather than generalized here (see Link 3). For
non-terminal tools, Link 2 is the only mitigation and its coverage is admittedly weak;
that residual is accepted (see Residual cost).

**If the prompt reframe stays flaky (fallback).** The critique already returns a
structured `verdict` enum but free-text `issues`. If "missing evidence" keeps
misfiring, replace the free-text issue with a structured enum of rejection reasons
(`MissingPrerequisite | DangerousAction | UnverifiableTarget | WrongTarget`), removing
the ambiguity that invites conflating vague narration with a missing prerequisite.
Deferred — try the prompt reframe first; this is the next lever if it underperforms.

**Related guard, not in scope.** A separate uncertainty-clarify guard
(`tool_prelude_phase.rs:~850`, gated by `policy_config.uncertainty_clarify_enforce`
+ `uncertainty_score >= threshold`) can *also* end a turn with a clarifying question
on a vague message + side-effecting tool, independent of the critique. It is not the
reported failure mode, but it produces a similar "vague message → dead end" UX.
Implementers should not assume Link 2 resolves all vague-message stalls; this guard is
a distinct path and is out of scope here.

**Residual cost (known and accepted).** Link 1 only covers terminal. For *non-terminal*
tools, the vague-message→missing-evidence conflation is mitigated only by the Link 2
prompt reframe (weak coverage) plus Link 4 cutting the confirmation driver. With Link 3
dropped, there is **no escape hatch** for a non-terminal cross-turn loop — but that loop
is currently **unobserved in telemetry**, so it is consciously left to Links 1/2/4 +
instrumentation. If telemetry later shows it, the event-log-derived design documented
under Link 3 closes it without schema changes.

### Link 3 — escape hatch after repeated identical blocks → **dropped from this spec**

An escape hatch keyed on a *consecutive-rejection counter* cannot work, because the
critique fires **at most once per user turn**: `set_plan_version`
(`tool_prelude_phase.rs:1007`) runs before the critique call, so on a rejection
`current_plan_version` is already set; the next iteration in the same turn hits the
`plan_already_generated` gate (`:975`) and skips the entire plan+critique block. The
observed deadlock is therefore **cross-turn** — the model reacts to the injected
"[SYSTEM] Critique pass blocked… re-plan / gather evidence" message by asking the user
to clarify, the user replies ("Run again", "Yes"), and each reply starts a *fresh*
`ExecutionState` (built per-turn in `main_loop.rs`) whose critique rejects again.

Consequence: an **intra-turn counter can never reach a threshold of 2** — it resets
every turn. Shipping it as "defense-in-depth" would be **dead code** that can't fire in
production, plus a unit test for an impossible scenario. So it is dropped, not shipped.

**Decision: ship no escape hatch in this spec.** Rationale:
- The *reported* incident is terminal + `Safe` and is fully resolved by **Link 1**.
- The cross-turn *drivers* are cut by **Link 4** (the "Yes" trap) and **Link 2** (fewer
  false rejections), so the loop is much less likely to form even for non-terminal tools.
- There is **no telemetry** of a non-terminal cross-turn critique loop. A tool-agnostic
  auto-bypass of a safety guard is exactly the speculative, durable machinery a
  self-hosted open-source agent should not carry until data justifies it.

**Documented design — implement only when telemetry shows the non-terminal cross-turn
loop in the wild** (kept here so the analysis isn't lost):
- *Count from already-persisted, immutable state, not a new counter.* At turn start,
  scan recent `ExecutionCritiquePass`-rejected events for the session matching
  `(tool, target_hint)` — the decision-point payload (`:1170`) already records
  `tool` + `target_hint` + `critique_result`. Zero new schema, append-only source,
  self-pruned by existing event machinery, testable by seeding events. Read from the
  **event store**, not the in-context message view (the sliding-window collapse strips
  old tool intermediates, so a context scan would miss prior blocks).
- *Conditional surfacing — never assert safety blindly.* Such a hatch consults only a
  repetition count, not risk, and would apply to all tools. For a non-terminal action
  (e.g. `edit_file` on a file never read — the *legitimate* groundedness block Link 2's
  rationale preserves) it must not both override the block and claim it "looks safe."
  Terminal + `classify_command == Safe` → confident copy and run; otherwise → neutral
  "this keeps blocking, I'll surface it rather than retry," and prefer surfacing over
  auto-running a possibly-legitimate block. Note: only terminal has a `needs_approval`
  backstop — non-terminal tools have none, another reason not to silently auto-run.

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
input — a persona leak that makes an internal guardrail look like an external tool
failure. Not fixed in this spec, but the concrete follow-up is cheap: append a system
instruction to the rejection payload injected at `tool_prelude_phase.rs:1190`, e.g.
*"(SYSTEM NOTE: this rejection came from your own internal safety guardrail, not the
external tool. Do not tell the user the tool failed or rejected the input.)"* so the
model stops attributing the block to the tool.

## Sequencing

Three fixes ship; leverage is uneven:

1. **Link 1** — highest leverage; likely resolves the reported `relevo --help` incident
   on its own. Land first.
2. **Link 4** — cheap, no LLM cost; fixes the "Yes"/"sure" trap. Land second.
3. **Link 2** — prompt-only change; low cost, uncertain benefit.
4. **Link 3** — **not implemented** (a counter-based hatch is dead code; the working
   cross-turn version is deferred until telemetry justifies it). No code lands.

## Testing

Project gate: `cargo fmt`, `cargo clippy --all-features -- -D warnings`, `cargo test`.

- **Link 1:** unit test — the terminal-Safe gating helper returns "skip" for
  `relevo --help` (Safe) and "run gating" for a Critical command (e.g. `rm -rf /`).
  Edge cases: `action: "check"` / malformed args / missing `command` → plan+critique
  still run (fail safe); a piped command with one non-Safe segment → still run.
- **Link 3:** no tests — nothing ships. (If the deferred event-log-derived hatch is
  built later, its test seeds N `ExecutionCritiquePass`-rejected events for
  `(tool, target_hint)` and asserts escape + conditional surfacing message.)
- **Link 4:** unit test — a bare confirmation passes through when the prior assistant
  turn matched the `"I still need the specific"` prefix; still intercepted on first
  occurrence. Exercise **both** nudge templates and more than one confirmation token
  ("yes", "sure", "go ahead").
- **Link 2:** prompt-text assertion that the new scoping instruction is present, plus
  an integration test (`MockProvider`) that a vague message + safe command is not
  blocked. Script the mock critique LLM to return `replan` + `missing evidence` on
  vague text and assert Link 1 still wins for a Safe terminal command. This link has
  inherently weaker automated coverage (LLM judgment); called out explicitly.
- **Composite acceptance (end-to-end):** the scenario that validates the shipped fixes
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
