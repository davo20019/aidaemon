# Pre-execution critique deadlock fix

**Date:** 2026-06-08
**Status:** Approved (design)
**Branch:** feature/sliding-window-phase0-observability

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
In `should_run_pre_execution_critique` (or its terminal-aware caller in
`tool_prelude_phase.rs`), when the tool call is `terminal`, run
`command_risk::assess()` on the command and **skip the critique when the assessment
is `Safe`**. Non-terminal tools keep current behavior. Reuses existing, well-tested
risk infra; generalizes to `ls`, `cat`, `which`, `--version`, `--help`, etc.

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

### Link 3 — escape hatch after repeated identical blocks
Track consecutive critique rejections keyed by `(tool_name, target_hint)`. After
**2 identical rejections**, the 3rd attempt bypasses the critique and the agent
surfaces the truth to the user instead of silently retrying, e.g.:
> "My groundedness check has refused `<command>` twice; it looks safe, so I'm running it now."

Then it proceeds. The counter resets when a different tool/target is attempted or the
action succeeds. Terminal's separate interactive approval flow (`needs_approval`)
still independently gates genuinely dangerous commands, so this cannot auto-run a
destructive command. N=2 mirrors the existing `verification_block_count` escape
pattern.

### Link 4 — anti-loop pass-through in the confirmation shortcut
Before re-issuing the canned "I still need the specific answer" nudge in
`maybe_handle_non_resolving_confirmation_shortcut`, check whether that *same* nudge
was the immediately preceding assistant turn. If it was, let the bare confirmation
fall through to the LLM instead of re-nudging. Breaks the loop after exactly one
nudge; preserves the helpful first nudge for genuine "X or Y" questions.

### Honesty note (related, lower priority)
The agent paraphrased its own internal critique block as if `relevo` had rejected the
input. Not addressed in this spec, but worth a follow-up: internal critique/guard
reasoning should not be presented to the user as an external tool's error.

## Testing

Project gate: `cargo fmt`, `cargo clippy --all-features -- -D warnings`, `cargo test`.

- **Link 1:** unit test — `should_run_pre_execution_critique` (terminal-aware path)
  returns `false` for `relevo --help` (Safe) and `true` for a Critical command
  (e.g. `rm -rf /`).
- **Link 3:** unit test on the rejection counter — escapes on the 3rd identical
  `(tool, target)` rejection; resets on a different target or on success.
- **Link 4:** unit test — bare "yes" passes through when the prior assistant turn was
  the specific-answer nudge; still intercepted on first occurrence.
- **Link 2:** prompt-text assertion that the new scoping instruction is present, plus
  an integration test (`MockProvider`) that a vague message + safe command is not
  blocked. This link has inherently weaker automated coverage (LLM judgment); called
  out explicitly.

## Out of scope
- Reworking `policy_bundle.risk_score` weighting.
- The honesty/sanitization follow-up above.
- Any change to terminal's interactive approval flow.
