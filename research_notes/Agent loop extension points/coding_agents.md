# Agent loop extension points in coding agents (Claude Code / Agent SDK, Codex CLI, Cline, Goose, Aider), as of 2026-11-08

Versions seen on 2026-11-08: Codex `rust-v0.167.0` (2026-11-06), Goose `v1.57.0` (2026-10-28), Cline `v4.1.26` (2026-10-26), Aider last tagged release `v0.86.0` (2025-08-09), all from the GitHub releases API. Claude Code docs cite versions up to v2.1.314. Repositories moved: Goose is now `aaif-goose/goose` (`block/goose` redirects). Codex docs moved from `developers.openai.com/codex/hooks` to `learn.chatgpt.com/docs/hooks` (308 redirect). Agent SDK docs moved from `platform.claude.com` to `code.claude.com/docs/en/agent-sdk/` (307 redirect).

## Which loop moments are hookable?

### Takeaway
Claude Code has the widest set of events (about 30) but no generic before/after model-call hook. Cline's SDK plugins are the only surface here with a mutating `beforeModel`. Codex has per-attempt model hooks that are diagnostic only. Goose has a small event set (12) with no compaction, model or subagent events. Aider has no hook system.

### Cited Findings

**Claude Code (CLI)**
- Run/session: `SessionStart` (matcher `startup|resume|clear|compact|fork`), `Setup`, `SessionEnd`, `UserPromptSubmit`, `UserPromptExpansion` — [Claude Code hooks reference](https://code.claude.com/docs/en/hooks).
- Tools: `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, and `PostToolBatch` (after a full parallel batch resolves, before the next model call) — [hooks reference](https://code.claude.com/docs/en/hooks).
- Model: no before/after-model-request event appears in the event list. The nearest are `PreModelSwitch`/`PostModelSwitch` (model changes only) and `MessageDisplay` (replaces on-screen text; the transcript and Claude keep the original) — [hooks reference](https://code.claude.com/docs/en/hooks).
- Model error and limits: `StopFailure` runs instead of `Stop` when a turn ends on an API error. Matcher values include `rate_limit`, `overloaded`, `billing_error`, `max_output_tokens`, `server_error`. Output and exit code are ignored except `terminalSequence`, so it is log/alert only — [hooks reference](https://code.claude.com/docs/en/hooks#stopfailure). There is no hook per retry attempt in the event list.
- Stop: `Stop` fires whenever Claude finishes responding, not only at task completion, and does not fire on user interrupts — [hooks guide](https://code.claude.com/docs/en/hooks-guide#limitations-and-troubleshooting).
- Budgets: hooks "may not fire when the agent hits the `max_turns` limit because the session ends before hooks can execute" — [Agent SDK hooks](https://code.claude.com/docs/en/agent-sdk/hooks).
- Compaction: `PreCompact` (can block; stdout becomes compaction instructions), `PostCompact` (receives `compact_summary`), and `SessionStart` with `compact` — [hooks reference](https://code.claude.com/docs/en/hooks).
- Subagents and teams: `SubagentStart`, `SubagentStop`, `TaskCreated`, `TaskCompleted`, `TeammateIdle`. A `Stop` in subagent frontmatter converts to `SubagentStop` — [hooks reference](https://code.claude.com/docs/en/hooks).
- Human in the loop: `PermissionRequest`, `PermissionDenied` (auto-mode denials), `Elicitation`/`ElicitationResult` (MCP input), `Notification`, and `PreToolUse` `defer` (pause and resume) — [hooks reference](https://code.claude.com/docs/en/hooks#defer-a-tool-call-for-later).
- Environment: `InstructionsLoaded`, `ConfigChange`, `CwdChanged`, `DirectoryAdded`, `FileChanged`, `WorktreeCreate`, `WorktreeRemove` — [hooks reference](https://code.claude.com/docs/en/hooks).

**Claude Agent SDK**
- In-process callbacks share the CLI event names, but availability differs. `PostToolBatch`, `UserPromptExpansion`, `MessageDisplay` and `StopFailure` are TypeScript-only — [Agent SDK hooks](https://code.claude.com/docs/en/agent-sdk/hooks#available-hooks).
- `SessionStart` and `SessionEnd` are callbacks in TypeScript only. In Python they exist only as shell hooks from settings files — [Agent SDK hooks](https://code.claude.com/docs/en/agent-sdk/hooks).

**Codex CLI**
- Events: `SessionStart`, `SessionEnd`, `SubagentStart`, `SubagentStop`, `UserPromptSubmit`, `PreToolUse`, `PermissionRequest`, `PostToolUse`, `PreCompact`, `PostCompact`, `PreModelRequest`, `PostModelRequest`, `Stop`, `Interrupt`. `Interrupt` and `SessionEnd` do not run for subagents — [Codex hooks](https://learn.chatgpt.com/docs/hooks).
- `PreModelRequest`/`PostModelRequest` run per attempt, including retries within a turn. `PostModelRequest` adds `status` (`completed|failed|cancelled`), `error`, `token_usage`. There is no `PostToolUseFailure` event in the list — [Codex hooks](https://learn.chatgpt.com/docs/hooks).
- Coverage gaps by design: hosted tools such as WebSearch get neither tool hook; `write_stdin` does not re-run `PreToolUse`; "Some specialized tool paths can opt out… Treat tool hooks as a useful guardrail, not a complete enforcement boundary." — [Codex hooks, Tool coverage](https://learn.chatgpt.com/docs/hooks).
- Code-mode nested tool calls do get hooks — [Codex hooks](https://learn.chatgpt.com/docs/hooks).

**Cline (current SDK path, v4.x)**
- Plugin hooks: `beforeRun`, `afterRun`, `beforeModel`, `afterModel`, `beforeTool`, `afterTool`, `onEvent` — [Cline SDK plugins](https://docs.cline.bot/sdk/plugins).
- SUPERSEDED: the v3.36-era hook docs and pages (`docs.cline.bot/features/hooks`, `/customization/hooks`) are now stubs pointing to SDK Plugins — [Cline hooks stub](https://docs.cline.bot/customization/hooks). The file-hook model (`TaskStart`, `PreToolUse`, etc.) was introduced in [v3.36](https://cline.bot/blog/cline-v3-36-hooks).
- Legacy file hooks still exist but are partly wired. In VS Code, `TaskStart` and `UserPromptSubmit` map to `beforeRun`, `PreToolUse` to `beforeTool`, `PostToolUse` to `afterTool`, and `TaskComplete`/`TaskCancel` to `afterRun`. "Deferred hooks (NOT wired here): TaskResume, TaskError, SessionShutdown, PreCompact, Notification." — [hooks-adapter.ts](https://github.com/cline/cline/blob/main/apps/vscode/src/sdk/hooks-adapter.ts).
- `afterModel.requestId` identifies "the final surfaced step, not every hidden HTTP retry" — [Cline SDK ARCHITECTURE](https://github.com/cline/cline/blob/main/sdk/ARCHITECTURE.md). The docs list no dedicated model-error hook — [Cline SDK plugins](https://docs.cline.bot/sdk/plugins).
- Compaction: the plugin examples list "custom compaction" — [Cline SDK plugins](https://docs.cline.bot/sdk/plugins). Subagents inherit compaction through session config, "not through a separate agent-level compaction hook surface" — [ARCHITECTURE](https://github.com/cline/cline/blob/main/sdk/ARCHITECTURE.md).

**Goose**
- Events: `SessionStart`, `SessionEnd`, `Stop`, `UserPromptSubmit`, `PreToolUse`, `PreToolUseResult`, `PostToolUse`, `PostToolUseFailure`, `BeforeReadFile`, `AfterFileEdit`, `BeforeShellExecution`, `AfterShellExecution`. Hooks ship in plugins (Open Plugins spec) — [Goose hooks guide](https://github.com/aaif-goose/goose/blob/main/documentation/docs/guides/context-engineering/hooks.md).
- "`SubagentStart` and `SubagentStop` are not currently emitted by goose" — [Goose hooks guide](https://github.com/aaif-goose/goose/blob/main/documentation/docs/guides/context-engineering/hooks.md).
- SUPERSEDED: PR #7411 described PreCompact/PostCompact and global-only config, but it [closed without merging](https://github.com/aaif-goose/goose/pull/7411). The current event table has no compaction events.
- Rust embedder surface: `goose-agent` exposes `PreToolCallHook<S>` (`Allow | Deny { reason }`), `PostToolCallHook<S>`, and `StopHook<S>` as `Operation`s on the state machine — [pre_tool_call_hook.rs](https://github.com/aaif-goose/goose/blob/main/crates/goose-agent/src/pre_tool_call_hook.rs), [stop_hook.rs](https://github.com/aaif-goose/goose/blob/main/crates/goose-agent/src/stop_hook.rs).

**Aider**
- No lifecycle hook system. A feature request for one is open: [#5712](https://github.com/Aider-AI/aider/issues/5712). [#2557](https://github.com/Aider-AI/aider/issues/2557) (command hooks) is closed.
- Nearest points are CLI options: `--lint-cmd`/`--auto-lint`, `--test-cmd`/`--auto-test`, `--notifications-command`, `--git-commit-verify`, `--commit-prompt`, `--load` — [Aider options](https://aider.chat/docs/config/options.html). Lint and test failures feed an auto-fix step — [Aider lint/test](https://aider.chat/docs/usage/lint-test.html).

### Inferences
- A kernel with typed before/after-model hooks plus a model-failure hook covers ground none of these fully do. Cline mutates model requests but has no documented failure hook. Codex sees failures per attempt but cannot act. Claude Code sees only terminal failures.
- A transcript-edit hook is essentially absent. Claude Code's `MessageDisplay` is display-only, and compaction hooks influence or observe summaries. Cline's `beforeModel` returning `messages` is the closest, and it changes the request (whether it persists is unconfirmed).

### Gaps
- I found no documented budget/limit hook (turns, tokens, cost) in any product other than the `max_output_tokens`/`billing_error` matchers on Claude Code's `StopFailure`, and Codex's `token_usage` observation.
- I could not confirm whether Cline's `beforeModel` message changes persist to session history.

## What may a hook do at each moment?

### Takeaway
All four hook-bearing products can deny a tool call or stop a run. Claude Code, Codex and Goose can also force a stopping turn to continue; I found no documented equivalent in Cline. Rewriting is uneven: Claude Code and Codex rewrite tool input, Cline also replaces tool results and model requests, and Goose hook files can only allow or deny.

### Cited Findings

**Claude Code**
- `PreToolUse`: `permissionDecision` `allow|deny|ask|defer`, `updatedInput` (replaces the whole input; permission rules evaluate the rewritten input), `additionalContext` — [hooks reference](https://code.claude.com/docs/en/hooks#pretooluse-decision-control).
- `allow` does not skip every check. Actions no mode auto-approves and network-path reads still prompt — [hooks reference](https://code.claude.com/docs/en/hooks#pretooluse-decision-control).
- `PostToolUse`: `decision: block` plus `reason`, `updatedToolOutput`, `additionalContext`; it cannot undo the action — [hooks reference](https://code.claude.com/docs/en/hooks), [hooks guide](https://code.claude.com/docs/en/hooks-guide#limitations-and-troubleshooting).
- `UserPromptSubmit` can reject and add context but cannot rewrite the prompt — [hooks reference](https://code.claude.com/docs/en/hooks).
- `Stop`/`SubagentStop`: `decision: block` continues the turn; `additionalContext` continues it under the same loop protections, labelled `Stop hook feedback` — [hooks reference](https://code.claude.com/docs/en/hooks#stop).
- `PermissionDenied`: `retry: true` tells the model it may retry. Claude Code "doesn't reverse the denial itself" — [hooks reference](https://code.claude.com/docs/en/hooks#permissiondenied-decision-control).
- Universal: `continue: false` stops Claude entirely and overrides event decision fields — [hooks reference](https://code.claude.com/docs/en/hooks).
- Handler types: `command`, `http`, `mcp_tool`, `prompt`, `agent` (agent is marked experimental) — [hooks reference](https://code.claude.com/docs/en/hooks).
- Pause/resume: `defer` works only in `-p` and only with a single tool call in the turn; with several calls it is ignored with a warning. The turn ends with `stop_reason: "tool_deferred"` — [hooks reference](https://code.claude.com/docs/en/hooks#defer-a-tool-call-for-later).
- Async hooks cannot block or modify. `asyncRewake` wakes Claude on exit 2 — [hooks reference](https://code.claude.com/docs/en/hooks), [Agent SDK hooks](https://code.claude.com/docs/en/agent-sdk/hooks).

**Codex**
- `PreToolUse`: deny (JSON or exit 2), `additionalContext`, and rewrite via `permissionDecision: "allow"` plus `updatedInput`. `ask`, legacy `approve`, `continue: false`, `stopReason`, `suppressOutput` are "parsed but not supported yet", and returning them marks the hook failed and **continues the tool call** — [Codex hooks](https://learn.chatgpt.com/docs/hooks).
- `PermissionRequest`: allow or deny; `updatedInput`, `updatedPermissions`, `interrupt` are reserved and "fail closed today" — [Codex hooks](https://learn.chatgpt.com/docs/hooks).
- `PostToolUse`: `decision: block` replaces the tool result with feedback and continues; `continue: false` also replaces the result — [Codex hooks](https://learn.chatgpt.com/docs/hooks).
- `Stop`: `decision: block` creates a continuation prompt from `reason`; `continue: false` ends. `PreCompact` `continue: false` stops before compacting — [Codex hooks](https://learn.chatgpt.com/docs/hooks).
- Model hooks and `Interrupt` cannot change or prevent anything: "Hook output is diagnostic only" — [Codex hooks](https://learn.chatgpt.com/docs/hooks).
- Handlers: `command` and `mcp_tool` run; `prompt` and `agent` are parsed but skipped — [Codex hooks](https://learn.chatgpt.com/docs/hooks).

**Cline**
- `beforeRun`: `stop`, `appendContext`. `beforeModel`: replace `messages`, `tools`, `options`, or `stop`. `afterModel`: `stop`. `beforeTool`: `skip` plus reason (goes to the model), replace `input`, `policy` (`autoApprove: false` forces approval), `appendContext`, `stop`. `afterTool`: replace `result`, `appendContext`, `stop`. `afterRun`, `onEvent`: nothing — [Cline SDK plugins](https://docs.cline.bot/sdk/plugins).
- File hooks can only return `cancel` (which becomes a whole-run stop), so one call cannot be refused while the task continues. Reported in [#14697](https://github.com/cline/cline/issues/14697).
- Cline docs recommend hooks for observation, and modifying through `beforeRun`/`beforeModel` — [Writing plugins](https://docs.cline.bot/sdk/guides/writing-plugins).

**Goose**
- "`PreToolUse` and `Stop` can block"; every other event ignores block decisions — [Goose hooks guide](https://github.com/aaif-goose/goose/blob/main/documentation/docs/guides/context-engineering/hooks.md).
- Denial text sent to the model is framework-fixed and includes "Do not retry" — [Goose hooks guide](https://github.com/aaif-goose/goose/blob/main/documentation/docs/guides/context-engineering/hooks.md).
- No input rewrite or context injection is documented for hook files. Non-blocking context injection is requested in [#10358](https://github.com/aaif-goose/goose/issues/10358), and tool results in `PostToolUse` payloads in [#13168](https://github.com/aaif-goose/goose/issues/13168). `emit_collecting_banners` collects a `banner` field for display — [hooks/mod.rs](https://github.com/aaif-goose/goose/blob/main/crates/goose/src/hooks/mod.rs).
- `PreToolUseResult` reports the chain's resolved `decision`, `policy_evaluated`, `blocked_by`, and `cause` (`policy_denial` or `hook_failure`). It is awaited, carries no authority, and is "best effort and not durable" — [Goose hooks guide](https://github.com/aaif-goose/goose/blob/main/documentation/docs/guides/context-engineering/hooks.md).

**Aider**
- Commands can trigger lint/test repair only; none can block or mutate a loop step beyond that — [Aider lint/test](https://aider.chat/docs/usage/lint-test.html).

### Inferences
- Goose's `PreToolUseResult` is a useful pattern to copy: a separate, authority-free record of what the chain decided and why, distinguishing policy denial from hook failure.
- Codex's "unsupported field means failed hook, tool continues" is a sharp edge: a hook that believes it is stopping a tool can fail open by returning an unimplemented field.

### Gaps
- I could not confirm the current return shape of Claude Code's `updatedInput` when two parallel hooks both rewrite; the docs state precedence for decisions only.

## How do multiple hooks compose?

### Takeaway
Claude Code and Codex launch matching hooks concurrently and resolve by restrictiveness. Goose runs hooks sequentially and stops at the first denial. Cline merges layers and short-circuits on stop.

### Cited Findings
- Claude Code: matching hooks run in parallel. With more than 12 prompt/agent hooks, 12 run at once and each waiting hook's timeout starts when it starts. An identical handler in several settings files runs once; plugin and skill copies stay separate — [hooks reference](https://code.claude.com/docs/en/hooks).
- Claude Code precedence: `deny > defer > ask > allow` on `PreToolUse`; `deny > ask > allow` on `PreModelSwitch`. Exit 2 routes like `deny` — [hooks reference](https://code.claude.com/docs/en/hooks#pretooluse-decision-control).
- Agent SDK: "Because completion order is non-deterministic, write each hook to act independently" — [Agent SDK hooks](https://code.claude.com/docs/en/agent-sdk/hooks).
- Configuration merges across levels; user, project and local hooks add to managed hooks rather than replacing them — [hooks reference](https://code.claude.com/docs/en/hooks).
- Codex: matching command hooks "are launched concurrently, so one hook can't prevent another matching hook from starting". Higher-precedence layers do not replace lower ones; all matching hooks load — [Codex hooks](https://learn.chatgpt.com/docs/hooks).
- Codex `PermissionRequest`: "any `deny` wins. Otherwise, an `allow` lets the request proceed." `Stop`: any `continue: false` beats continuation decisions from other hooks — [Codex hooks](https://learn.chatgpt.com/docs/hooks).
- Codex background hooks: at most eight concurrently per session; may finish out of order — [Codex hooks](https://learn.chatgpt.com/docs/hooks).
- Goose: `emit_blocking_with_outcome` iterates matched rules in a `for` loop, awaiting each, and returns at the first denial. Later hooks do not run — [hooks/mod.rs](https://github.com/aaif-goose/goose/blob/main/crates/goose/src/hooks/mod.rs). There is no allow-beats-deny path, because allow simply moves on.
- Cline file hooks: controls merge with `cancel`/`review` OR-ed and reasons joined — [hook-file-hooks.ts](https://github.com/cline/cline/blob/main/sdk/packages/core/src/hooks/hook-file-hooks.ts). "A stopping layer ends the merge: later layers' hooks are not run" — same file.
- Cline plugins are bound at session start; edits affect only later sessions — [Cline SDK plugins](https://docs.cline.bot/sdk/plugins).

### Inferences
- The two models trade differently. Parallel-most-restrictive gives bounded latency but makes hook-to-hook input transformation ill-defined. Sequential-first-deny gives deterministic chaining but sums latency.
- A kernel with ordered hooks has a stronger composition story than any of the parallel products, provided ordering is explicit and a deny cannot be overridden later.

### Gaps
- Codex's documented `PreToolUse` precedence when hooks disagree (for example two different `updatedInput`s) was not found.
- Cline's plugin-hook execution order across multiple plugins is not in the docs. Only the merge code for layered hooks was inspected.

## What happens when a hook crashes or times out?

### Takeaway
Fail-open is the default almost everywhere, with opt-in fail-closed now added by Claude Code and Goose. The notable split is Claude Code itself: an SDK callback timeout on `PreToolUse` blocks, while a command hook timeout does not.

### Cited Findings
- Claude Code command hooks: exit 0 success; exit 2 blocks on blocking events; other exit codes are non-blocking errors — [hooks reference](https://code.claude.com/docs/en/hooks).
- Claude Code default timeouts: 600s for `command`/`http`/`mcp_tool`, reduced to 30s on `UserPromptSubmit` and model-switch events and 10s on `MessageDisplay`; `prompt` 30s; `agent` 60s; `SessionEnd` shares a 1.5s budget (up to 60s) — [hooks reference](https://code.claude.com/docs/en/hooks), [hooks guide](https://code.claude.com/docs/en/hooks-guide#limitations-and-troubleshooting).
- A timed-out `command`, `http` or `mcp_tool` `PreToolUse` hook "doesn't block the tool call" unless `onFailure: "block"` is set — [hooks reference](https://code.claude.com/docs/en/hooks).
- `onFailure` (v2.1.295+): `continue` or `block`; no effect on `Stop`, `SubagentStop`, `TaskCompleted`, `TeammateIdle`, or background hooks — [hooks reference](https://code.claude.com/docs/en/hooks).
- Agent SDK timeout by event: `PreToolUse` **blocks** the call; `UserPromptSubmit` blocks the prompt; `PreModelSwitch` blocks the switch; `Stop`/`SubagentStop` count as no decision; `Notification`, `PreCompact`, and others log and continue — [Agent SDK hooks](https://code.claude.com/docs/en/agent-sdk/hooks#hook-timeout).
- Agent SDK throw: the action goes ahead with a `hook_callback_failed` message; "A `PreToolUse` guard that crashes then lets the tool call reach the normal permission flow"; returning nothing also counts as failure. `onFailure: "block"` needs TS SDK v0.3.317+ or Python v0.2.177+ — [Agent SDK hooks](https://code.claude.com/docs/en/agent-sdk/hooks).
- History: before v2.1.273, a timed-out `Stop` callback discarded other hooks' decisions; before v2.1.208, interrupting during a pending `PreToolUse` callback could let the tool proceed — [Agent SDK hooks](https://code.claude.com/docs/en/agent-sdk/hooks#hook-timeout).
- Codex timeouts: 600s default; `SessionEnd` and `Interrupt` 1s, up to 3s — [Codex hooks](https://learn.chatgpt.com/docs/hooks).
- Codex failure: "A `PreToolUse` error, timeout, or malformed response can let the tool run"; MCP errors, missing servers and unavailable tools do not block — [Codex hooks](https://learn.chatgpt.com/docs/hooks).
- Cline plugin hooks: 3s timeout. Default `hookErrorMode: "ignore"` continues; `"throw"` fails the run. Five consecutive failures turn a `degraded` plugin `failed` — [Cline SDK plugins](https://docs.cline.bot/sdk/plugins). `beforeTool` throwing "will count as a tool failure" per [Writing plugins](https://docs.cline.bot/sdk/guides/writing-plugins), which sits awkwardly with the `ignore` default.
- Cline isolation: each plugin gets a worker thread per session, "crash isolation, not a security boundary"; hosts without the bootstrap fail closed with `PluginBootstrapMissingError` — [Cline SDK plugins](https://docs.cline.bot/sdk/plugins).
- Cline file hooks: blocking tool hook timeout defaults to 120s; start hooks are fire-and-forget unless `blockingRunStartHooks` is set — [hook-file-hooks.ts](https://github.com/cline/cline/blob/main/sdk/packages/core/src/hooks/hook-file-hooks.ts), [subprocess.ts](https://github.com/cline/cline/blob/main/sdk/packages/core/src/hooks/subprocess.ts).
- Goose: 30s default. Failure means logged and continue. `on_failure: block` applies to `PreToolUse` only; `Stop` "stays fail-open even with `on_failure: block`" — [Goose hooks guide](https://github.com/aaif-goose/goose/blob/main/documentation/docs/guides/context-engineering/hooks.md). Requested in [#10866](https://github.com/aaif-goose/goose/issues/10866).
- Goose's strict decision protocol treats stray stdout, invalid UTF-8, and failed payload delivery as no decision, because "repairing them can turn malformed output into JSON that parses as an allow". A malformed matcher or config still fails open — [Goose hooks guide](https://github.com/aaif-goose/goose/blob/main/documentation/docs/guides/context-engineering/hooks.md).

**Stop-loop guards**
- Claude Code: `stop_hook_active` input plus an 8-consecutive-continuation cap, reset when Claude calls a tool; `CLAUDE_CODE_STOP_HOOK_BLOCK_CAP` raises it — [hooks reference](https://code.claude.com/docs/en/hooks#stop).
- Goose: `DEFAULT_DENY_LIMIT = 8`, per turn in the operation builder; `GOOSE_STOP_HOOK_BLOCK_CAP` overrides — [stop_hook.rs](https://github.com/aaif-goose/goose/blob/main/crates/goose-agent/src/stop_hook.rs), [services.rs](https://github.com/aaif-goose/goose/blob/main/crates/goose/src/agents/services.rs).
- Codex: `stop_hook_active` input is documented — [Codex hooks](https://learn.chatgpt.com/docs/hooks).

### Inferences
- Goose's rule that failure on a stop hook may never trap a finished turn, and Claude Code's identical exemption, suggest that fail-closed should be a property of the event, not a global switch.

### Gaps
- I found no documented continuation cap for Codex `Stop` hooks.

## Is hook/run state durable across restarts?

### Takeaway
Hooks are essentially stateless. Durability, where it exists, belongs to the session transcript, and Claude Code's `defer` is the only documented durable hook-driven pause.

### Cited Findings
- Claude Code `defer`: "There is no timeout or retry limit. The session stays on disk until the `cleanupPeriodDays` retention sweep deletes it, after 30 days by default." On resume the hook may decide or `defer` again — [hooks reference](https://code.claude.com/docs/en/hooks#defer-a-tool-call-for-later).
- Claude Code `SessionStart` resumes with `source` matcher `resume`; skill-frontmatter hooks last for the session; direct settings edits are "normally picked up automatically by the file watcher" — [hooks reference](https://code.claude.com/docs/en/hooks).
- Codex: trust is persisted against a hook hash. Background hooks are cancelled at session end and undelivered output is discarded — [Codex hooks](https://learn.chatgpt.com/docs/hooks).
- Goose: `PreToolUseResult` delivery is "best effort and not durable: … no record is kept if the hook does not run" — [Goose hooks guide](https://github.com/aaif-goose/goose/blob/main/documentation/docs/guides/context-engineering/hooks.md).
- Cline: session-bound; plugins' state dies with the worker/session, and `ctx.onDispose` handles cleanup — [Cline SDK plugins](https://docs.cline.bot/sdk/plugins).

### Inferences
- A kernel that persists pending hook interrupts (an approval awaiting an owner) and can resume across a daemon restart exceeds what these products document outside Claude Code's `-p` defer path.

### Gaps
- I found no documentation of persisted hook counters, such as whether Claude Code's or Goose's stop-block counts survive restart. Probably not, but unconfirmed.
- I found no documentation of a Codex or Goose hook-driven pause/resume.

## Trust model: who may install hooks, and how are repository hooks handled?

### Takeaway
Codex has the strongest default: non-managed hooks need hash-pinned review. Claude Code gates interactive sessions with one folder-trust dialog but explicitly runs repository hooks in `-p` and SDK runs. Goose and Cline load project-scoped hooks with no trust gate that I could find.

### Cited Findings
- Claude Code interactive: hooks from every settings file are held until the trust dialog is accepted. `-p` or SDK: "never shows the dialog and treats the folder as trusted, so hooks committed in a repository's `.claude/settings.json` run in a folder you've never trusted" — [hooks reference](https://code.claude.com/docs/en/hooks#workspace-trust).
- Hooks and the `env` block are used even when only a parent folder was trusted; project subagent frontmatter hooks are not — [Permissions](https://code.claude.com/docs/en/permissions#what-runs-before-you-trust-a-folder).
- `disableAllHooks` in user settings is not enough, because project settings can set it back to `false`; use `--settings '{"disableAllHooks": true}'`, `--setting-sources user`, or `--bare` — [Permissions](https://code.claude.com/docs/en/permissions#what-runs-before-you-trust-a-folder).
- Claude Code enterprise: `allowManagedHooksOnly` blocks user, project, local and plugin hooks; `disableAllHooks` cannot disable managed hooks except from managed settings; `allowedHttpHookUrls` limits HTTP targets — [hooks reference](https://code.claude.com/docs/en/hooks).
- The docs say `if` filters are best-effort, so "use the permission system rather than a hook to enforce a hard allow or deny" — [hooks reference](https://code.claude.com/docs/en/hooks).
- Codex: project hooks load only if the `.codex/` layer is trusted. "Before a non-managed hook can run, Codex requires you to review and trust the exact hook definition", recorded by hash; new or changed hooks are skipped until trusted. Plugin hooks use the same flow. `--dangerously-bypass-hook-trust` skips it — [Codex hooks](https://learn.chatgpt.com/docs/hooks).
- Codex managed hooks from system, MDM, cloud or `requirements.toml` are trusted by policy and cannot be disabled in the browser. `allow_managed_hooks_only` exists — [Codex hooks](https://learn.chatgpt.com/docs/hooks).
- Goose: project plugins under `<project>/.agents/plugins/` "are loaded when goose is started from that project"; the docs only warn "Only install or create hooks from sources you trust" — [Goose hooks guide](https://github.com/aaif-goose/goose/blob/main/documentation/docs/guides/context-engineering/hooks.md). The discovery code loads project plugins when given a `project_root`, and a search of `discovery.rs` found no trust check — [discovery.rs](https://github.com/aaif-goose/goose/blob/main/crates/goose/src/plugins/discovery.rs). SUPERSEDED: PR #7411's global-only claim did not survive.
- Cline: project plugins at `.cline/plugins/` are "available only when working in that project" — [Cline plugins](https://docs.cline.bot/customization/plugins). Hook search paths include workspace `.cline/hooks` and a deprecated directory — [paths.ts](https://github.com/cline/cline/blob/main/sdk/packages/shared/src/storage/paths.ts). Enabled by default unless a session `plugins` policy turns them off, and that policy is not applied in `sandbox` mode — [Cline SDK plugins](https://docs.cline.bot/sdk/plugins).
- Cline tool names not listed in `toolPolicies` "default to enabled and auto-approved" — [Permission handling](https://docs.cline.bot/sdk/guides/permission-handling).

### Inferences
- Codex's hash-pinned review is the closest match to a design where repository content cannot grant itself authority. Claude Code's headless exception is the exact shape of a daemon's exposure.
- For a daemon, the relevant lesson from `-p`/SDK is that unattended modes have no one to show a trust prompt to, so the safe default is that repository hooks stay off rather than that they run.

### Gaps
- Could not confirm whether Goose or Cline have any trust prompt outside the inspected files and docs; absence in those is not proof of absence in UI code.

## What problems have users reported?

### Takeaway
Three failure classes recur across products: hooks that silently do not fire (headless, subagents, mid-session), stop hooks that fight the permission system or the user, and trust state that the agent itself can reach. Issues below are user reports, and closure does not imply a fix.

### Cited Findings

**Claude Code**
- Advisories: [GHSA-q5hj-mxqh-vv77 / CVE-2026-40068](https://github.com/anthropics/claude-code/security/advisories/GHSA-q5hj-mxqh-vv77) (high): a spoofed git worktree `commondir` bypassed the trust dialog and ran repository hooks (affected >=2.1.63, <2.1.84). [GHSA-ff64-7w26-62rf / CVE-2026-25725](https://github.com/anthropics/claude-code/security/advisories/GHSA-ff64-7w26-62rf) (high): sandboxed code could create a missing `.claude/settings.json` and inject hooks that ran with host privileges at restart (fixed v2.1.2).
- Not firing: [#88738](https://github.com/anthropics/claude-code/issues/88738) (open, 2026-10): `PreToolUse` stopped mid-session and never recovered. [#34692](https://github.com/anthropics/claude-code/issues/34692): subagent tool calls did not fire hooks (v2.1.76; closed NOT_PLANNED). [#33343](https://github.com/anthropics/claude-code/issues/33343): hooks not enforced in `-p` (closed NOT_PLANNED, and references earlier reports closed as stale). I could not confirm whether either of these two was later fixed; current SDK docs do document `agent_id` on hooks firing inside subagents.
- Stop loops: [#98066](https://github.com/anthropics/claude-code/issues/98066) (open): a `/goal` Stop hook re-fired about 15 times after the auto-mode classifier denied the only remaining step. [#61156](https://github.com/anthropics/claude-code/issues/61156) (closed): hook-created task files re-triggered the hook. [#77105](https://github.com/anthropics/claude-code/issues/77105) (closed NOT_PLANNED): a goal Stop hook overrode a live user stop instruction.
- Note that #98066's roughly 15 re-fires exceed the documented cap of 8 consecutive continuations, but the cap resets on each tool call — [hooks reference](https://code.claude.com/docs/en/hooks#stop).

**Codex**
- [#37890](https://github.com/openai/codex/issues/37890) (open): in full-access mode, an agent created a persistent global hook and approved it through a nested TUI by supplying terminal input.
- [#21639](https://github.com/openai/codex/issues/21639) (open): hooks stopped running after a Desktop update. [#24211](https://github.com/openai/codex/issues/24211) (open): user-level `PostToolUse` ignored while plugin `PostToolUse` ran.
- [#56793](https://github.com/openai/codex/issues/56793) (open): `.git` metadata in a plugin's install changes the trust hash, forcing re-review with no real hook change.
- Repository advisories listed none involving hooks; the one returned concerned a sandbox path bug ([GHSA-w5fx-fh39-j5rw](https://github.com/openai/codex/security/advisories/GHSA-w5fx-fh39-j5rw)).

**Cline**
- [#13380](https://github.com/cline/cline/issues/13380) (closed): `hook.afterRun` capability request had no timeout, hanging sessions.
- [#14031](https://github.com/cline/cline/issues/14031) (open): Linux `TaskStart` hook stays running and `contextModification` is not applied. [#13822](https://github.com/cline/cline/issues/13822) (open): failing to inject context from `TaskStart`.
- [#11944](https://github.com/cline/cline/issues/11944) (open): plugin sandbox bootstrap not bundled in VS Code 4.0.0, so plugins showed installed but never ran. [#14697](https://github.com/cline/cline/issues/14697) (open) also says plugins do not load on Desktop or VS Code, which is the reporter's claim.
- [#11098](https://github.com/cline/cline/issues/11098) (closed): task restarted after abort during hook execution.
- Advisories found were not hook-specific: [GHSA-3cj3-hqcr-g934](https://github.com/cline/cline/security/advisories/GHSA-3cj3-hqcr-g934), [GHSA-5c57-rqjx-35g2](https://github.com/cline/cline/security/advisories/GHSA-5c57-rqjx-35g2) (WebSocket hijacking).

**Goose**
- [#12746](https://github.com/aaif-goose/goose/issues/12746) (closed): `PreToolUse` hooks loaded but never ran in headless `goose run`. [#12650](https://github.com/aaif-goose/goose/issues/12650) (closed): hooks did not run for Summon subagent tool calls.
- [#13186](https://github.com/aaif-goose/goose/issues/13186) (open): denying `recipe__final_output` loops the subagent because a continuation prompt overrides "Do not retry" (92 denied calls over 8 minutes in one run, subagents defaulting to `GOOSE_MAX_TURNS` 1000).
- [#11272](https://github.com/aaif-goose/goose/issues/11272) (closed RFC): an allow is not a bounded, verifiable authorization at the effect point.
- Docs bug: [#12178](https://github.com/aaif-goose/goose/issues/12178) (closed): guide examples used tool names the CLI never sends. A bare `"*"` matcher is an invalid regex and the rule is "silently skipped" — [Goose hooks guide](https://github.com/aaif-goose/goose/blob/main/documentation/docs/guides/context-engineering/hooks.md).
- Hook-adjacent advisory: [GHSA-r5pp-p5r8-466r / CVE-2026-72718](https://github.com/aaif-goose/goose/security/advisories/GHSA-r5pp-p5r8-466r) (high): `goose review` command execution via git `core.fsmonitor`. It concerns git config, not the hook system.

**Aider**
- [#5376](https://github.com/Aider-AI/aider/issues/5376), [#5057](https://github.com/Aider-AI/aider/issues/5057), [#3583](https://github.com/Aider-AI/aider/issues/3583): commits use `--no-verify` by default, bypassing local git pre-commit hooks. The default is confirmed by `--git-commit-verify` default False — [Aider options](https://aider.chat/docs/config/options.html).

### Inferences
- Silent non-firing is the most dangerous recurring class, because a policy hook's failure looks like a pass. Goose's `policy_evaluated` field and Claude Code's per-hook debug lines are attempts at observability for this.
- #98066, #77105 and Goose #13186 share one structure: a continuation or completion mechanism pressing against a denial. Denial needs to bind the continuation layer, not only the tool call.

### Gaps
- GitHub search coverage was keyword-limited and rate-limited, so the issue lists are samples rather than a census. Codex "hook not firing" searches with long queries returned nothing, while shorter ones did.
- Reproduction and fix status for individual issues was not verified beyond their state and closure reason.
