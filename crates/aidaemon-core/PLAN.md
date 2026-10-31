# aidaemon SDK: plan

This is the working plan for turning aidaemon into a framework people can
build their own agents with. It is written so that any later session, human
or agent, can pick up where the last one stopped. `DESIGN.md` explains *why*
the kernel looks the way it does; this file says *what is done, what is next,
and how to resume*.

Keep it current: when a step finishes, tick it, add the commit, and move the
"Next up" marker.

## Where things stand

| Item | State |
|---|---|
| Branch | `ccr-986102ee-12nt48` (branched from `master` at `4a36e32`) |
| Kernel crate | `crates/aidaemon-core`, experimental, `publish = false` |
| Tests | 30 behavioral + 8 unit + 1 doc test, all passing |
| CI | Lint, Linux tests, release build, coverage green. macOS job never gets a runner (see "Known issues") |
| PR | #13 was closed without merging; the branch was kept. Open a new PR only when asked |
| Research | Harness Field Guide: https://claude.ai/artifact/Fm5Ax7oX2vzmACjtb7iPxF (14 concepts, compared across ~20 harnesses, with verdicts) |

### What exists in the kernel today

Agent builder, `Provider` (+ `Retry` layer), simple `Tool` with conservative
annotations, `ToolContext` (trusted ids beside the arguments), `Hook`
(`before_model`, `before_tool`, `after_tool`, `on_stop`, `on_stuck`),
`Receipt` for every call, typed `RunOutcome` / `RunStatus`, serializable
`RunState` with interrupt/resume, `Event` stream, `RunStore` +
`MemoryStore`, hard `Limits`, custom `StopCondition`, configurable stuck
detection, read-only calls in parallel, a `testing` feature with
`ScriptedProvider`, and `examples/quickstart.rs`.

## Decisions already made

1. **Same repo, new crates.** The SDK grows as workspace crates in this repo,
   so it can reuse `harness_eval`, providers and tools, and so the daemon
   becomes its first user. No separate repository.
2. **Small kernel, everything else is a hook.** If a feature cannot be built
   on the public hooks, add a hook point; do not add a kernel primitive.
3. **New loop, not an extraction.** `src/agent/loop/` (~62k lines) is not
   moved into the SDK. Daemon policies are re-implemented as hooks one at a
   time.
4. **Port only what measures well.** A daemon policy is ported only if
   repeated live runs (N-of-M pass rates, not single runs) show it helps.
5. **Receipts are the source of truth** for "did the work happen"; completion
   checks read receipts, not the model's prose.

## Open questions (decide before Phase 1)

These shape the public API. Each has a current default in the sketch.

- [ ] **Tool errors.** `Tool::call` returns `ToolResult`, and any error goes
      back to the model; it never ends the run. Keep?
- [ ] **Errors vs statuses.** Provider outages and limits are
      `Ok(RunOutcome { status })`, `Err` is only misuse and store failures.
      Keep?
- [ ] **Approvals.** Pausing for a human is a hook decision (`Interrupt`),
      not a separate approval API. Keep?
- [ ] **Instructions.** The system prompt is re-sent every request and never
      stored in `RunState`. Keep?
- [ ] **Naming.** `aidaemon-core` vs `aidaemon-sdk` for the user-facing
      crate (a facade crate could re-export the others behind features).

## Roadmap

Each phase ends in something runnable and tested. Phases 1-3 can overlap.

### Phase 0: lock the primitives  ← Next up

- [ ] Review `src/` of the kernel against the Field Guide verdicts.
- [ ] Answer the open questions above; update code and `DESIGN.md`.
- [ ] Decide whether to merge the sketch to `master` now (as experimental)
      or keep it on the branch until Phase 1.

Done when: the five questions are answered and the API reflects them.

### Phase 1: a real model

- [ ] New crate `crates/aidaemon-providers` with an OpenAI-compatible
      provider (reuse wire logic from `src/providers/openai_compatible.rs`,
      drop daemon-only options such as llama.cpp slot pinning into its own
      config).
- [ ] Anthropic provider with prompt caching on the stable prefix.
- [ ] Streaming: add a `Provider::stream` method yielding typed deltas;
      `complete` becomes a fold over it; matching `Event` variants.
- [ ] Run `examples/quickstart.rs` against a real model behind an env var.

Done when: an example agent does a small real task end to end with a real
provider, and streaming events reach an `EventSink`.

### Phase 2: tools worth having

- [ ] New crate `crates/aidaemon-tools`: read file (bounded views),
      exact-match edit with uniqueness check, write, search (glob/grep),
      shell, web fetch, checklist (`track_requirements`).
- [ ] Apply the agent-computer-interface lessons by default: paginated and
      capped output, actionable errors, validate edits before applying.
- [ ] `ExecutionBackend` trait for containment: local, macOS Seatbelt,
      Linux bubblewrap or Landlock, Docker. Keep it separate from approval
      policy (the Codex two-axis model).
- [ ] Opt-in `TypedTool` trait for typed effects and access manifests, with
      a blanket adapter to `Tool`.

Done when: a coding agent built only from SDK crates can read, edit and test
a small synthetic repo inside a sandbox.

### Phase 3: durability

- [ ] New crate `crates/aidaemon-store-sqlite` implementing `RunStore`, with
      versioned migrations (a version table, not best-effort `ALTER`s).
- [ ] Crash-recovery test: kill mid-run, `load_state` + `run_from`, finish.
- [ ] Idempotency keys (`run_id + tool + canonical args hash`) and per-tool
      replay decisions, so a redone turn does not repeat side effects.

Done when: a run survives a process restart without repeating a
non-idempotent tool call.

### Phase 4: measurement

- [ ] Add a `core` engine to `src/harness_eval` and run the existing 16 YAML
      fixtures against `aidaemon-core`.
- [ ] Live eval runner: the same scenarios against a real model, N runs
      each, reporting pass rates.
- [ ] Baseline the plain kernel (no daemon policies) so later hooks have
      something to beat.

Done when: there is a single command that prints pass rates for the old loop
and the new kernel side by side.

### Phase 5: port daemon policies as hooks

One policy per change, each gated by Phase 4 numbers. Suggested order:

- [ ] Approvals and risk levels → `before_tool` (`Interrupt` / `Deny`).
- [ ] Receipt-based completion check and checklist → `on_stop`.
- [ ] Untrusted-output sanitization and secret redaction → `after_tool`.
- [ ] Project instructions (AGENTS.md / CLAUDE.md resolver) → `before_model`
      plus a `before_tool` deferral when entering a new subtree.
- [ ] Context strategy (turn eviction, summaries, spill-to-file) →
      `before_model`, task state kept outside the transcript.
- [ ] Memory injection → `before_model`, in `crates/aidaemon-memory`.

Done when: each ported policy shows a measured gain or is dropped.

### Phase 6: the rest of the framework

- [ ] `crates/aidaemon-agents`: `Agent::as_tool()`, explicit context forking,
      child budgets that count toward the parent, structured results.
- [ ] `crates/aidaemon-mcp`: MCP client exposing servers as `Tool`s, with
      lazy schema loading.
- [ ] OpenTelemetry `EventSink` (GenAI semantic conventions).
- [ ] Move `testing` into its own crate if it grows.
- [ ] Public docs and a second example (a non-coding agent).

### Phase 7: the daemon runs on the SDK

- [ ] Daemon builds its agent through `AgentBuilder` with its hooks.
- [ ] Old loop runs side by side behind a flag until the new one matches or
      beats it on Phase 4 numbers.
- [ ] Delete the old loop.

## Parallel track: fixes found in the daemon

Independent of the SDK; each is a small PR against `master`.

- [ ] `src/tools/cli_agent.rs:662-711` launches external agents with their
      safety off (`--dangerously-skip-permissions`,
      `--dangerously-bypass-approvals-and-sandbox`, `--yolo`). Default to
      sandboxed / permission modes, make full bypass opt-in.
- [ ] No default sandbox on Linux (Docker backend is behind an empty,
      off-by-default `execution-docker` feature).
- [ ] `PlanRecovery::recover_interrupted_plans` (`src/plans/recovery.rs:39`)
      is only called from tests.
- [ ] `is_non_delegable_tool` (`src/mandates/mod.rs:11`) checks an `mcp__`
      prefix, but MCP tools are named `{server}__{tool}`. Add a test.
- [ ] State migrations have no version table.

## Known issues

- **macOS CI job never runs.** `Test (macos-14)` is cancelled within a second
  with no runner assigned (`runner_id: 0`) on every push since 2026-10-31.
  Master passed on macOS on 2026-10-23. Likely Actions billing or a spending
  limit for macOS minutes; check GitHub Settings → Billing and plans →
  Actions. Not caused by code.

## Working agreements

- Commits are authored as **David Loor M. <davo20019@gmail.com>**. No
  `Co-Authored-By: Claude` or session trailers in commits, and no
  "Generated with Claude Code" footer in PR descriptions.
- A new cloud container defaults git to Claude's identity. Either add
  `git config --global user.name/user.email` to the environment's setup
  script, or run them at the start of a session.
- Follow the repo's `CLAUDE.md`: `cargo fmt`, `cargo clippy --all-features
  -- -D warnings`, and the relevant tests before committing. For this crate:
  `cargo clippy -p aidaemon-core --all-targets --all-features -- -D warnings`
  and `cargo test -p aidaemon-core`.
- Synthetic data only in tests and docs (no real names, IDs or PII).

## How to resume

```bash
git fetch origin ccr-986102ee-12nt48
git checkout ccr-986102ee-12nt48
git config user.name "David Loor M."
git config user.email "davo20019@gmail.com"
cargo test -p aidaemon-core
cargo run -p aidaemon-core --example quickstart --features testing
```

Prompt to start a new session:

> Continue the aidaemon SDK work. Check out branch `ccr-986102ee-12nt48`,
> read `crates/aidaemon-core/PLAN.md` and `DESIGN.md`, then work on the
> step marked "Next up". Commit as David Loor M. with no Claude attribution.
