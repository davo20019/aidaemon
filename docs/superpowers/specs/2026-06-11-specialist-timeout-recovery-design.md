# Specialist Timeout Recovery Design

## Problem

When a synchronous `spawn_agent` call times out, the tool currently returns a
successful `Ok(String)` containing timeout text. The agent loop can therefore
treat the call as progress and accept a long follow-up response that:

- exposes internal orchestration details;
- presents a plan instead of the requested deliverable; and
- promises unsupported future behavior such as monitoring the connection or
  retrying later.

The existing deferred-action detector recognizes phrases such as "I will
retry", but the substantive-text bypass can accept the response because the
remaining plan text is long enough.

## Desired Behavior

After a synchronous specialist timeout or specialist execution error:

1. The result is classified as a failed tool attempt.
2. The parent receives an explicit recovery instruction to stop relying on the
   failed delegation and continue with direct tools when feasible.
3. The parent must not claim it is monitoring infrastructure or will retry in
   the future unless a real background task or retry has been scheduled.
4. A status report or proposed plan is not accepted as the final answer when
   the requested deliverable is still absent.
5. If direct completion is impossible, the final answer reports the concrete
   blocker and completed evidence without inventing future work.

Background `spawn_agent` calls are unchanged: a successfully queued background
task may truthfully state that its result will be delivered later.

## Design

### Timeout And Error Classification

Keep the existing `Tool` result interface, but make synchronous `spawn_agent`
failure strings use established error prefixes:

- `Error: specialist timed out after ...`
- `Error: specialist failed: ...`

This lets `classify_tool_result_failure_with_context` recognize the attempt as
a failure without introducing a tool-specific out-of-band status channel.
Timeouts classify as transient; explicit child errors use the existing text
classifier.

### Recovery Directive

When result learning observes a failed `spawn_agent` call, append a
tool-result notice that instructs the parent to:

- pivot to available direct tools;
- avoid another unchanged delegation attempt;
- complete as much of the original request as possible now; and
- avoid statements about monitoring, connection stability, or future retries
  unless an actual background task exists.

The generic transient cooldown remains in force. The new notice adds
delegation-specific behavior rather than replacing cooldown accounting.

### Completion Quality

Add a focused response-analysis predicate for incomplete status/plan replies.
It should identify responses that combine:

- an execution failure or inability statement;
- future-action language such as retrying, monitoring, waiting, or continuing
  later; and
- plan/status scaffolding such as "Current Plan", "Research Phase", or
  numbered future phases.

The predicate must use word- or phrase-boundary matching appropriate to natural
language and must not reject:

- a completed answer that includes a brief methodology section;
- a concrete partial result with evidence plus an honest blocker;
- a truthful acknowledgement of a queued background task; or
- ordinary informational answers containing the word "plan".

When the predicate matches, it overrides the substantive-text bypass and uses
the existing deferred-response recovery path. After recovery is exhausted, the
agent should return a concrete blocker or evidence summary rather than the
status/plan prose.

## Data Flow

1. `spawn_agent` executes synchronously and times out or returns a child error.
2. The tool returns a prefixed failure string.
3. Tool execution classifies and records the failure.
4. Result learning appends transient cooldown guidance and the
   delegation-pivot notice.
5. The next LLM iteration uses direct tools or produces a concrete blocker.
6. Completion analysis rejects unsupported retry/status prose even when it is
   long.

## Testing

Add regression coverage for:

1. Synchronous specialist timeout text classifies as transient failure.
2. Synchronous specialist child errors classify as failures.
3. Failed `spawn_agent` results receive the direct-tool pivot notice.
4. The exact reported response pattern is recognized as incomplete despite its
   length.
5. A completed briefing with a methodology or next-steps section remains
   substantive.
6. A successfully queued background delegation acknowledgement remains valid.

Run focused tests first, then the repository pre-commit checks:

```bash
cargo fmt
cargo clippy --all-features -- -D warnings
cargo test
```

## Scope

This change does not add automatic retry scheduling, alter specialist timeout
values, or redesign background task delivery. It corrects failure signaling,
parent recovery behavior, and final-response acceptance.
