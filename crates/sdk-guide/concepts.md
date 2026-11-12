# Coming from other SDKs

A rough map. The ideas line up closely; the differences are noted.

| You know | In aidaemon | Notes |
|---|---|---|
| Claude Agent SDK: `query()` / agent loop | `Agent::run(session_id, message)` | The loop is in `aidaemon-core`; model-agnostic (`Provider` trait). |
| Claude Agent SDK: custom tools, MCP servers | `Tool` trait / `FnTool`; `aidaemon-mcp` (`McpServer::tools()`) | MCP tools become ordinary `Tool`s. |
| Claude Agent SDK: hooks (PreToolUse, PostToolUse, Stop) | `Hook::before_tool`, `after_tool`, `on_stop` | Hooks can `Deny`, `Rewrite`, or `Interrupt`. |
| Claude Agent SDK: permission modes, `canUseTool` | `ApprovalPolicy` + `ApprovalMode`, or your own `before_tool` | A paused run is a serializable `RunState`, resumed with `Agent::resume`. |
| Claude Agent SDK: subagents | `aidaemon-agents` (agent as a tool) | Child budgets roll up into the parent. |
| Claude Agent SDK: sessions | session id passed to `run` + a `RunStore` (`aidaemon-store-sqlite`) | |
| Vercel AI SDK: `generateText` / `streamText` | `Provider` (streaming) used by the loop | You normally call `Agent::run`, not the model directly. |
| Vercel AI SDK: `tool({ description, parameters, execute })` | `ToolSpec { name, description, parameters }` + `Tool::call` | `parameters` is JSON Schema. A `#[tool]` macro is planned. <!-- TODO(dx): switch to #[tool] --> |
| Vercel AI SDK: `maxSteps`, `stopWhen` | `Limits` (`Agent::builder().limits(..)`), `StopCondition` | Also token budgets and stuck detection. |
| Vercel AI SDK: provider packages | `aidaemon-providers` (`Anthropic`, `OpenAiCompatible`, ...) | |
| LangGraph: graph of nodes and edges | the loop plus hooks | No graph DSL: control flow is the loop; add behaviour in hooks. |
| LangGraph: state, checkpointer | `RunState` (serde) + `RunStore` | Checkpoints are written as the run goes; crash recovery resumes in-flight tool batches. |
| LangGraph: `interrupt()` / human in the loop | `ToolDecision::Interrupt` then `Agent::resume` | Interrupts happen before a tool call. |
| LangGraph: long-term memory store | `aidaemon-memory` (`MemoryStore`, `Embedder`) | Swappable; plugs in as a hook. |
| LangSmith / tracing | `Event` stream (`.event_sink`), `policies` tracing (`otel` feature) | |
| Evals | `aidaemon-eval`: YAML scenarios, N-of-M pass rates | |

Where aidaemon differs on purpose: every tool call produces a receipt (what
ran, with what arguments, how it ended), completion checks can read receipts
instead of trusting the model's prose, and tool output is treated as untrusted
unless a tool says otherwise.
