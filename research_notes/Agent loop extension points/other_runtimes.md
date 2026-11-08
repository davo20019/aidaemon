# Agent loop extension points: Microsoft Agent Framework / Semantic Kernel, Vercel AI SDK, Mastra, OpenHands Software Agent SDK

State as of 2026-11-08. Versions checked with `gh release list`: Agent Framework python-1.25.0 and dotnet-1.28.0; Semantic Kernel dotnet-1.82.0 and python-1.47.0; Mastra `@mastra/core` 1.82.0; OpenHands Software Agent SDK v1.60.2 (source read at that tag); OpenHands app v1.37.0; AI SDK packages released 2026-11-07, with `ai` 7.0.162 the newest version I saw cited (in an issue).

How to read this:
- Microsoft Learn pages were read in full. AI SDK, Mastra and OpenHands doc pages were read through a summarizing fetcher, so exact wording may differ. Load-bearing claims were cross-checked in source where a source link is given.
- Source links to `main` were read on 2026-11-08.
- GitHub issue statements are reporter claims unless I say I confirmed them in source.

## Which loop moments are hookable in each runtime?

### Takeaway
Microsoft Agent Framework has the fullest surface: an onion middleware at run, model and tool level, plus a separate experimental fail-closed "Agent Hooks" control plane with eight named points. Mastra has the most model-adjacent stages and is the only one here with a typed API-error hook. AI SDK splits into three mutating seams (`prepareStep`, `toolApproval`, model middleware) and a set of observe-only callbacks. OpenHands has six hook events and no model-call hook; condenser, stuck detector and security analyzer are separate built-in subsystems.

### Cited Findings

**Semantic Kernel (superseded; still receiving releases)**
- The README calls Agent Framework "the enterprise-ready successor to Semantic Kernel" — [SK README](https://github.com/microsoft/semantic-kernel/blob/main/README.md).
- There are exactly three filter types: Function Invocation, Prompt Render, Auto Function Invocation — [SK filters](https://learn.microsoft.com/en-us/semantic-kernel/concepts/enterprise-readiness/filters).
- Function Invocation fires for every `KernelFunction`, prompt-based or method-based. Prompt Render fires only for prompt rendering, not method functions. Auto Function Invocation fires only inside automatic function calling and adds chat history, the list of functions to run, and iteration counters — [SK filters](https://learn.microsoft.com/en-us/semantic-kernel/concepts/enterprise-readiness/filters).
- With `IChatCompletionService` used directly, filters run only if a `Kernel` is passed — [SK filters](https://learn.microsoft.com/en-us/semantic-kernel/concepts/enterprise-readiness/filters).
- Automatic invocation is capped by a constant, `MaximumAutoInvokeAttempts = 128` — [FunctionCallsProcessor.cs](https://github.com/microsoft/semantic-kernel/blob/main/dotnet/src/InternalUtilities/connectors/AI/FunctionCalling/FunctionCallsProcessor.cs).

**Microsoft Agent Framework: middleware (GA API)**
- Three chained types. .NET: Agent Run, Function calling, `IChatClient`. Python: agent, function, chat. Go: agent-level `agent.Middleware`, plus provider and function middlewares — [MAF middleware](https://learn.microsoft.com/en-us/agent-framework/agents/middleware/).
- Python chat middleware runs inside the function loop, once per model call, including calls that send tool results back — [MAF middleware](https://learn.microsoft.com/en-us/agent-framework/agents/middleware/).
- .NET function middleware works only with agents using `FunctionInvokingChatClient`, for example `ChatClientAgent` — [MAF middleware](https://learn.microsoft.com/en-us/agent-framework/agents/middleware/).
- Limits are configuration, not hooks: `max_iterations` (default 40), `max_function_calls`, `max_duration_seconds`, `max_consecutive_errors_per_request` (default 3), `terminate_on_unknown_calls` — [_tools.py](https://github.com/microsoft/agent-framework/blob/main/python/packages/core/agent_framework/_tools.py).
- Compaction is a separate experimental subsystem. A `CompactionProvider` registered on `ChatClientBuilder` runs inside the tool loop before each LLM call. Registered through `ChatClientAgentOptions`, it does not engage during the tool loop — [MAF compaction](https://learn.microsoft.com/en-us/agent-framework/agents/conversations/compaction).
- Python takes `compaction_strategy`, an async callable that mutates a message list. The Python harness has separate before and after phases, and `AgentTriggeredCompactionStrategy` lets the model request compaction through a tool — [MAF compaction](https://learn.microsoft.com/en-us/agent-framework/agents/conversations/compaction).

**Microsoft Agent Framework: Agent Hooks (experimental, Python only)**
- Implements the framework-neutral AGENT-HOOKS-0.1 contract. It is explicitly "a control plane, not a telemetry plane"; not available in .NET or Go — [Agent Hooks](https://learn.microsoft.com/en-us/agent-framework/agents/agent-hooks).

| Interception point | When | Transform target |
| --- | --- | --- |
| `agent_startup` | Before first input in a hooks session | None |
| `input` | External request enters | Input content, role |
| `pre_model_call` | Before each model request | Messages |
| `post_model_call` | After each complete response | Content, framework-executed tool calls, finish reason |
| `pre_tool_call` | Before each framework-executed tool | Arguments |
| `post_tool_call` | After success or failure | Result |
| `output` | Before final response reaches caller | Final content |
| `agent_shutdown` | Completed, failed or cancelled | None |

Table source: [Agent Hooks](https://learn.microsoft.com/en-us/agent-framework/agents/agent-hooks).
- Provider-hosted tools appear only in `post_model_call`; `pre_tool_call` and `post_tool_call` cannot block them — [Agent Hooks](https://learn.microsoft.com/en-us/agent-framework/agents/agent-hooks).
- Each nested agent needs its own bundle for its internal model and tool activity — [Agent Hooks](https://learn.microsoft.com/en-us/agent-framework/agents/agent-hooks).

**Vercel AI SDK**
- Agent classes: `ToolLoopAgent`, `HarnessAgent` — [Agents overview](https://ai-sdk.dev/docs/agents/overview); `WorkflowAgent` from `@ai-sdk/workflow` — [WorkflowAgent](https://ai-sdk.dev/docs/agents/workflow-agent).
- Callbacks on `ToolLoopAgent`: `onStart`, `onStepStart`, `onToolExecutionStart`, `onToolExecutionEnd`, `onStepEnd`, `onStepError`, `onError`, `onAbort`, `onEnd`. `onStepFinish` and `onFinish` are deprecated aliases — [ToolLoopAgent reference](https://ai-sdk.dev/docs/reference/ai-sdk-core/tool-loop-agent).
- `onStepError` fires when a step fails before producing a result; `onStepEnd` then does not fire. Provider retries stay inside one step (`maxRetries`) — [ToolLoopAgent reference](https://ai-sdk.dev/docs/reference/ai-sdk-core/tool-loop-agent).
- Per-tool input hooks: `onInputStart`, `onInputDelta` (`streamText` only), `onInputAvailable` — [Tool calling](https://ai-sdk.dev/docs/ai-sdk-core/tools-and-tool-calling).
- `prepareCall` runs per call; `prepareStep` runs before each step; `repairToolCall` handles invalid tool calls — [ToolLoopAgent reference](https://ai-sdk.dev/docs/reference/ai-sdk-core/tool-loop-agent), [Tool calling](https://ai-sdk.dev/docs/ai-sdk-core/tools-and-tool-calling).
- Model middleware has `transformParams`, `wrapGenerate`, `wrapStream` — [Language model middleware](https://ai-sdk.dev/docs/ai-sdk-core/middleware).
- Stop conditions: `isStepCount`, `hasToolCall`, `isLoopFinished`, or a custom `StopCondition`. `ToolLoopAgent` defaults to `isStepCount(20)`; `WorkflowAgent` has no default step limit — [Loop control](https://ai-sdk.dev/docs/agents/loop-control).
- Steering is experimental (`experimental_createSteering`, `experimental_steering`) — [Loop control](https://ai-sdk.dev/docs/agents/loop-control).
- Subagents are ordinary tools calling another agent's `generate()` or `stream()` — [Subagents](https://ai-sdk.dev/docs/agents/subagents).

**Mastra**
- Processor methods: `processInput` (once before the loop), `processInputStep` (each step), `processLLMRequest` (after conversion to provider prompt), `processLLMResponse`, `processOutputStream` (per chunk), `processOutputStep`, `processOutputResult`, `processAPIError` (API rejection such as 400 or 422) — [Processors](https://mastra.ai/docs/agents/processors).
- Loop options: `maxSteps`, `stopWhen`, `prepareStep`, `onStepFinish`, `onFinish`, `onError`, `onAbort`, `onIterationComplete`, `isTaskComplete`, `maxProcessorRetries`, `modelSettings.maxRetries` — [Agent.generate reference](https://mastra.ai/reference/agents/generate).
- Subagent hooks: `delegation.onDelegationStart`, `delegation.onDelegationComplete`, `delegation.messageFilter` — [Agent.generate reference](https://mastra.ai/reference/agents/generate).
- HITL: `requireToolApproval` per call, `requireApproval` per tool, in-tool `suspend()`, and `resumeStream` / `approveToolCall` / `declineToolCall` — [Agent approval](https://mastra.ai/docs/agents/agent-approval).
- Context limits: `TokenLimiter` and `ToolCallFilter` processors — [Processors](https://mastra.ai/docs/agents/processors); Observational Memory's Observer and Reflector — [Observational Memory](https://mastra.ai/docs/memory/observational-memory).
- Workflows pause at step level with `suspend()`, resume with `run.resume({ step, resumeData })` — [Suspend and resume](https://mastra.ai/docs/workflows/suspend-and-resume).

**OpenHands Software Agent SDK**
- The closed hook enum: `PreToolUse`, `PostToolUse`, `UserPromptSubmit`, `SessionStart`, `SessionEnd`, `Stop` — [hooks/types.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/hooks/types.py).
- Hook kinds: `command` (default), `prompt`, `agent` — [Hooks guide](https://docs.openhands.dev/sdk/guides/hooks).
- `PreToolUse`, `PostToolUse` and `UserPromptSubmit` run from an event callback when `ActionEvent`, `ObservationEvent` and user `MessageEvent` are emitted — [conversation_hooks.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/hooks/conversation_hooks.py).
- `Stop` runs only when status is `FINISHED`, with `reason="agent_finished"`. `PAUSED` and `STUCK` break out before it; the iteration and budget limits emit `MaxIterationsReached` / `MaxBudgetReached` errors with no hook — [local_conversation.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/conversation/impl/local_conversation.py).
- `max_iteration_per_run` defaults to 500 — [local_conversation.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/conversation/impl/local_conversation.py).
- The condenser runs inside `Agent._step` before the LLM call. If it returns a `Condensation`, the step emits that event and returns without calling the model — [agent.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/agent/agent.py).
- Model errors are handled in `_step`, not by hooks: `LLMContextWindowExceedError` and `LLMMalformedConversationHistoryError` emit a `CondensationRequest`, the latter after `state.rebuild_view()`, when the condenser handles requests — [agent.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/agent/agent.py).
- The stuck detector is configurable only by on/off and thresholds: `action_observation=4`, `action_error=3`, `monologue=3`, `alternating_pattern=6` — [conversation/types.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/conversation/types.py).
- It nudges once on an action-error streak, otherwise sets `STUCK` — [local_conversation.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/conversation/impl/local_conversation.py).
- Risk assessment and confirmation policy are separate abstractions in the control loop — [SDK paper v2 §4](https://arxiv.org/html/2511.03690v2).

### Inferences

| Moment | SK filters (legacy) | MAF middleware | MAF Agent Hooks (py, exp.) | AI SDK | Mastra | OpenHands SDK |
| --- | --- | --- | --- | --- | --- | --- |
| Run start / end | No dedicated filter | Agent/run wrap | `agent_startup`/`input`, `output`/`agent_shutdown` | `onStart`, `onEnd` | `processInput`, `processOutputResult`, `onFinish` | `SessionStart`/`SessionEnd`, `UserPromptSubmit` |
| Before model | Prompt Render, prompt functions only | Chat wrap | `pre_model_call` | `prepareStep`, `onStepStart`, `transformParams` | `processInputStep`, `processLLMRequest` | None |
| Wrap model | None | Chat/`IChatClient` wrap | Via bundle internals only | `wrapGenerate`/`wrapStream` | No wrap-style processor found | None |
| After model | None | Chat wrap | `post_model_call` | `onStepEnd` (after tools too) | `processOutputStep`, `processLLMResponse` | None |
| Model error / retry | Function filter retry (function scope) | Chat wrap can catch | `post_model_call` is success-only as documented | `onStepError`, `onError`; retry has no callback | `processAPIError`, `abort(retry)` | Built-in `_step` branches only |
| Before / after tool | Function and Auto filters | Function wrap | `pre_tool_call`, `post_tool_call` | `toolApproval`, `onToolExecutionStart/End` | Approval; no tool wrap found in docs read | `PreToolUse`, `PostToolUse` |
| Stop can be refused | No | No | `output` can deny, not extend | No | `onIterationComplete` feedback, `isTaskComplete`, `processOutputStep` retry | `Stop` deny resumes |
| Limit reached hook | None | None found | None listed | Custom `StopCondition` decides; no reached-limit event | `stopWhen`; `TokenCostControl` is a processor | None |
| Compaction | Not a filter | Separate strategies, no hook point | None | `prepareStep` messages | Processors, Observational Memory | Condenser subsystem, no hook |
| Subagents | Not covered | Per-agent | Per-agent bundle | None | `delegation.*` | Not confirmed |
| HITL | Not a filter feature | Tool approval, run returns | `Verdict.escalate` + resolver | `toolApproval`, run returns | Suspend/approve/resume | `WAITING_FOR_CONFIRMATION` |

- In OpenHands, a tool error emitted as `AgentErrorEvent` and a hook rejection emitted as `UserRejectObservation` appear not to fire `PostToolUse`, since the dispatcher checks `isinstance(event, ObservationEvent)`. Not tested at runtime.
- OpenHands' `PreToolUse` runs when the action is emitted, so it appears to run before the confirmation pause, not after a human decides.
- AI SDK's retry wrapper calls the model's `doGenerate` ([generate-text.ts](https://github.com/vercel/ai/blob/main/packages/ai/src/generate-text/generate-text.ts)), so `wrapGenerate` should run once per attempt. Not tested at runtime.

### Gaps
- I found no documented per-attempt retry callback in AI SDK, MAF or OpenHands. Absence in the pages read is not proof.
- I could not confirm whether OpenHands hooks apply inside delegated or `TaskTool` subagents.
- OpenHands' `critic` package (present in the source tree) was not examined.
- Mastra's `prepareStep` return contract is not on the reference page read.

## What may a hook do at each moment: observe, mutate, block, short-circuit, retry?

### Takeaway
Three designs appear. Wrap-style middleware (SK, MAF, AI SDK model middleware) can do anything by controlling `next`. Verdict-style hooks (Agent Hooks, OpenHands) return a constrained decision. Staged processors (Mastra) mutate data and signal through `abort`. Retry is first-class only in Mastra, and an explicit "refuse to stop" only in OpenHands' `Stop`, with Mastra's `onIterationComplete` and retry path as a softer equivalent.

### Cited Findings

**Semantic Kernel**
- Function filter: inspect arguments, handle exceptions, override the result before or after execution, retry, for example on an alternative model. Not calling `next` prevents execution — [SK filters](https://learn.microsoft.com/en-us/semantic-kernel/concepts/enterprise-readiness/filters).
- Prompt filter: view and modify the prompt, or prevent submission by overriding the result — [SK filters](https://learn.microsoft.com/en-us/semantic-kernel/concepts/enterprise-readiness/filters).
- Auto filter: set `Terminate`. In sequential mode the loop returns after the terminating function; with concurrent invocation all tasks complete and all results are added before stopping — [FunctionCallsProcessor.cs](https://github.com/microsoft/semantic-kernel/blob/main/dotnet/src/InternalUtilities/connectors/AI/FunctionCalling/FunctionCallsProcessor.cs).

**Microsoft Agent Framework**
- Middleware may inspect and modify input and output, skip `call_next`, or set `context.result` and raise `MiddlewareTermination` — [MAF middleware](https://learn.microsoft.com/en-us/agent-framework/agents/middleware/).
- .NET function middleware can set `FunctionInvocationContext.Terminate`. The docs warn this can leave function-call content without a result, making history unusable — [MAF middleware](https://learn.microsoft.com/en-us/agent-framework/agents/middleware/).
- .NET non-streaming-only run middleware forces streaming calls to run non-streaming. `Use(sharedFunc:)` keeps streaming but cannot intercept output — [MAF middleware](https://learn.microsoft.com/en-us/agent-framework/agents/middleware/).
- Agent Hooks verdicts: `allow`, `deny`, `transform`, plus SDK helpers `Verdict.warn` and `Verdict.escalate`. `transform` paths must start at `$target`; a bad transform fails closed — [Agent Hooks](https://learn.microsoft.com/en-us/agent-framework/agents/agent-hooks).
- Run- and model-level denies raise `InterceptionBlocked`. A tool-seam policy deny instead returns a control error to the model without the denied payload, and the loop continues — [Agent Hooks](https://learn.microsoft.com/en-us/agent-framework/agents/agent-hooks).
- `evaluate_only` records verdicts without acting — [Agent Hooks](https://learn.microsoft.com/en-us/agent-framework/agents/agent-hooks).

**Vercel AI SDK**
- `prepareStep` can override `model`, `messages`, `toolChoice`, `activeTools`, sampling settings, `stopSequences`, `seed`, `reasoning`, `experimental_sandbox`. Settings apply to the current step, but a returned `messages` becomes the base for later steps — [Loop control](https://ai-sdk.dev/docs/agents/loop-control).
- `runtimeContext` and `toolsContext` may be updated in `prepareStep` — [ToolLoopAgent reference](https://ai-sdk.dev/docs/reference/ai-sdk-core/tool-loop-agent).
- `toolApproval` returns `'not-applicable'`, `'approved'`, `'denied'` or `'user-approval'`, optionally with a `reason`. A denial reason reaches the model as an `execution-denied` result; an approval reason does not. Provider-executed tools ignore it. `needsApproval` is deprecated — [Tool calling](https://ai-sdk.dev/docs/ai-sdk-core/tools-and-tool-calling).
- `wrapGenerate`/`wrapStream` can modify parameters, call the model and modify results — [Language model middleware](https://ai-sdk.dev/docs/ai-sdk-core/middleware).
- The loop also ends on a non-tool finish reason, a tool without `execute`, or a call needing approval — [Loop control](https://ai-sdk.dev/docs/agents/loop-control).

**Mastra**
- Mutation: `processInput`, `processInputStep`, `processOutputStep`, `processOutputResult`, `processOutputStream`; `processLLMRequest` changes are transient and not saved to history — [Processors](https://mastra.ai/docs/agents/processors).
- `processOutputStream` drops a chunk by returning `null`/`undefined` — [Processors](https://mastra.ai/docs/agents/processors).
- `abort()` throws `TripWire`. `abort(reason, { retry: true })` replays the step with the reason as a system reminder and requires `maxProcessorRetries`. `processAPIError` returns `{ retry: true }` after editing `messageList` — [Processors](https://mastra.ai/docs/agents/processors).
- `onIterationComplete` can return `continue: false` or `feedback` — [Agent.generate reference](https://mastra.ai/reference/agents/generate).
- `onDelegationStart` can modify or reject a delegation; `onDelegationComplete` can return `{ feedback }` and call `bail()` — [Agent.generate reference](https://mastra.ai/reference/agents/generate).

**OpenHands**
- Exit 2 or a `deny` blocks `PreToolUse`, `UserPromptSubmit` and `Stop`; `PostToolUse`, `SessionStart`, `SessionEnd` cannot block — [Hooks guide](https://docs.openhands.dev/sdk/guides/hooks).
- A blocked action is recorded in `blocked_actions`. The agent later pops it and emits `UserRejectObservation` with `rejection_source="hook"` instead of executing, so the tool call still gets a result — [conversation_hooks.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/hooks/conversation_hooks.py), [agent.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/agent/agent.py).
- `additional_context` is consumed for `UserPromptSubmit` (appended to `extended_content`), `SessionStart` (delivered after the system prompt) and `Stop` (as feedback). The `PreToolUse` handler does not read it — [conversation_hooks.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/hooks/conversation_hooks.py).
- A denied `Stop` injects feedback as an `environment` user message, sets status back to `RUNNING`, and continues — [local_conversation.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/conversation/impl/local_conversation.py).
- `HookDecision` has only `ALLOW` and `DENY`; `ASK` is commented out as future work — [hooks/types.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/hooks/types.py).

### Inferences
- AI SDK's lifecycle callbacks are effectively observe-only: the helper that invokes them ignores results and swallows errors ([notify.ts](https://github.com/vercel/ai/blob/main/packages/ai/src/util/notify.ts)). Mutation and blocking must go through `prepareStep`, `toolApproval`, or middleware.
- MAF's tool-seam deny and OpenHands' rejection event make the same choice: a refused tool call becomes a tool result so the model can adapt and the call/result pairing stays valid. MAF's .NET `Terminate` warning shows the cost of the alternative.
- OpenHands' `PreToolUse` cannot rewrite arguments in the code read; Agent Hooks can, which forces its re-approval rules (see the HITL findings below).

### Gaps
- Whether a throw from `prepareStep` or `wrapGenerate` is retried or surfaced is not documented on the AI SDK pages read.
- Whether OpenHands' `continue: false` output changes control flow beyond `should_continue` was not traced. The docs do not define it.

## How do multiple hooks compose: ordering, wrap versus sequential, which decision wins?

### Takeaway
Middleware composes as an onion, so outer position means both first look and last word. Verdict and processor systems run sequentially, where first deny wins. Agent Hooks is the only one with a closed, declared set of composition profiles. AI SDK stop conditions are an OR, and Mastra's error processors stop at the first retry.

### Cited Findings
- SK Python filters nest in registration order (`before 1, before 2, function, after 2, after 1`). In .NET, dependency-injection order is not guaranteed; use `Kernel` properties when order matters — [SK filters](https://learn.microsoft.com/en-us/semantic-kernel/concepts/enterprise-readiness/filters).
- MAF: agent-level wraps run-level, `A1 -> A2 -> R1 -> R2 -> Agent -> R2 -> R1 -> A2 -> A1`. Function and chat middleware follow the same principle — [MAF middleware](https://learn.microsoft.com/en-us/agent-framework/agents/middleware/).
- MAF Go: `Middlewares` apply in declared order, first is outermost — [MAF middleware](https://learn.microsoft.com/en-us/agent-framework/agents/middleware/).
- Agent Hooks default is sequential `first_deny` with approval stopping the fold, so controls that must always run belong before those that can request approval. One bundle per agent; stacking is rejected; middleware placed before it is outside the boundary — [Agent Hooks](https://learn.microsoft.com/en-us/agent-framework/agents/agent-hooks).
- AGENT-HOOKS-0.1 defines a closed profile set: `sequential/first_deny` (`on_approval: stop|resume`), `sequential/run_all`, `parallel/strictest`, `parallel/unanimous`. Knob defaults are normative and fail-closed — [spec](https://github.com/responsibleai/agent-hooks/blob/main/spec/0.1/AGENT-HOOKS-0.1.md).
- Under `on_approval: "stop"`, a permit ends the emission with `fold_truncated: true`, so skipped interceptors are recorded — [spec](https://github.com/responsibleai/agent-hooks/blob/main/spec/0.1/AGENT-HOOKS-0.1.md).
- AI SDK middleware arrays nest as `firstMiddleware(secondMiddleware(yourModel))` — [Language model middleware](https://ai-sdk.dev/docs/ai-sdk-core/middleware).
- AI SDK stop conditions all run through `Promise.all`, then `.some` — [stop-condition.ts](https://github.com/vercel/ai/blob/main/packages/ai/src/generate-text/stop-condition.ts).
- AI SDK constructor and per-call callbacks both run, constructor first, except constructor-only `onStepError`, `onError`, `onAbort` — [ToolLoopAgent reference](https://ai-sdk.dev/docs/reference/ai-sdk-core/tool-loop-agent).
- AI SDK `notify` fires a callback list in parallel — [notify.ts](https://github.com/vercel/ai/blob/main/packages/ai/src/util/notify.ts).
- Mastra processors run in array order. With memory: input is memory then yours; output is yours then memory. An output `abort()` skips memory, so messages are not saved — [Processors](https://mastra.ai/docs/agents/processors).
- Mastra error processors default to `ProviderHistoryCompat`, `PrefillErrorHandler`, `UnsupportedFileHandler`, `StreamErrorRetryProcessor`, stopping at the first `{ retry: true }` — [Processors](https://mastra.ai/docs/agents/processors).
- Mastra workflow-as-processor runs branches with `.parallel()`, and `.map()` selects which branch's output continues — [Processors](https://mastra.ai/docs/agents/processors).
- Mastra approval flags combine by OR; a tool-level function overrides the combined result — [Agent approval](https://mastra.ai/docs/agents/agent-approval).
- OpenHands hooks run sequentially via `execute_all`. `stop_on_block=True` applies to `PreToolUse`, `UserPromptSubmit`, `Stop`; the others run all — [executor.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/hooks/executor.py), [manager.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/hooks/manager.py).
- OpenHands `additional_context` values are joined with newlines. The first blocking result supplies the reason — [manager.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/hooks/manager.py).
- OpenHands async hooks cannot block, and a warning is logged if configured on `PreToolUse` — [manager.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/hooks/manager.py).
- OpenHands `PipelineCondenser` stops at the first condenser returning a `Condensation` — [pipeline_condenser.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/context/condenser/pipeline_condenser.py).
- OpenHands `EnsembleSecurityAnalyzer` takes max severity across children; `UNKNOWN` is filtered unless `propagate_unknown=True` — [ensemble.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/security/ensemble.py).
- OpenHands `ConfirmRisky` defaults to threshold `HIGH` with `confirm_unknown=True` — [confirmation_policy.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/security/confirmation_policy.py).

### Inferences
- In onion middleware, "which decision wins" is positional: the outermost layer can rewrite anything inside it. Agent Hooks addresses this by demanding first position ("treat outer position as outer trust") and refusing partial installation.
- The risk in first-deny with approval-stop is that a later hard policy is skipped once an approval permits. The spec's `fold_truncated` record and MAF's ordering advice both acknowledge it.

### Gaps
- OpenHands' docs do not state hook ordering. The source order is the concatenation of matcher lists in config order; merge precedence across plugin and project sources was not traced beyond `HookConfig.merge` existing.
- I did not find documentation of ordering between a Mastra `prepareStep` and `processInputStep`.

## What happens when a hook raises: fail open or fail closed?

### Takeaway
There is no consensus, and within one product the answer changes by mechanism. Only Agent Hooks is fail-closed by contract. AI SDK callbacks fail open silently. OpenHands policy hooks fail open while its security analyzer fails closed. MAF's plain function middleware converts exceptions to tool errors and needed a new `MiddlewareFailure` type to fail closed.

### Cited Findings
- **Agent Hooks:** a raised exception maps to `host_error:interceptor_failed`, a timeout to `host_error:interceptor_timeout`, a bad verdict to `host_error:verdict_invalid`, each a deny — [spec](https://github.com/responsibleai/agent-hooks/blob/main/spec/0.1/AGENT-HOOKS-0.1.md).
- **Agent Hooks:** timeout defaults to five seconds, but a synchronous interceptor that blocks the event loop cannot be preempted. Enforcement availability couples to agent availability "by design" — [Agent Hooks](https://learn.microsoft.com/en-us/agent-framework/agents/agent-hooks).
- **Agent Hooks:** a host or enforcement failure halts the run, unlike a policy deny at a tool seam — [Agent Hooks](https://learn.microsoft.com/en-us/agent-framework/agents/agent-hooks).
- **MAF function middleware:** ordinary exceptions become tool-error results and the loop continues, which the source calls "fail-open for enforcement layers". `MiddlewareFailure` is never converted, cancels the concurrent batch, and propagates; a synchronous tool in a worker thread may still finish its side effects — [_middleware.py](https://github.com/microsoft/agent-framework/blob/main/python/packages/core/agent_framework/_middleware.py).
- **MAF:** on service-managed conversations the loop first submits one error `function_result` per dangling call so the hosted thread stays valid — [_middleware.py](https://github.com/microsoft/agent-framework/blob/main/python/packages/core/agent_framework/_middleware.py).
- **MAF agent and chat middleware:** every exception propagates to the caller — [_middleware.py](https://github.com/microsoft/agent-framework/blob/main/python/packages/core/agent_framework/_middleware.py).
- **SK .NET auto invocation:** the invocation path catches `Exception` and returns `"Error: Exception while invoking function..."` as the function result — [FunctionCallsProcessor.cs](https://github.com/microsoft/semantic-kernel/blob/main/dotnet/src/InternalUtilities/connectors/AI/FunctionCalling/FunctionCallsProcessor.cs).
- **AI SDK callbacks:** "Errors in callbacks do not break the generation flow"; the code is an empty `catch {}` — [notify.ts](https://github.com/vercel/ai/blob/main/packages/ai/src/util/notify.ts); docs say tool execution callback errors are silently caught — [Tool calling](https://ai-sdk.dev/docs/ai-sdk-core/tools-and-tool-calling).
- **AI SDK tools:** errors from `execute` become `tool-error` parts and permit further roundtrips — [Tool calling](https://ai-sdk.dev/docs/ai-sdk-core/tools-and-tool-calling).
- **Mastra processors:** non-`TripWire` errors are rethrown in the output paths; `TripWire` invokes `onViolation` and ends processing — [runner.ts](https://github.com/mastra-ai/mastra/blob/main/packages/core/src/processors/runner.ts). Errors thrown in `onViolation` are caught — [Processors](https://mastra.ai/docs/agents/processors).
- **Mastra approval predicate:** if the `requireToolApproval` function throws, approval is required. On durable or stored agents a function value falls back to requiring approval — [Agent approval](https://mastra.ai/docs/agents/agent-approval).
- **OpenHands command hooks:** only exit 2 blocks. Other non-zero exits are logged and the operation proceeds — [Hooks guide](https://docs.openhands.dev/sdk/guides/hooks).
- **OpenHands:** timeout, missing command and other exceptions return `success=False` without `blocked` — [executor.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/hooks/executor.py).
- **OpenHands prompt and agent hooks:** fail to `allow`, with `success=False` so the fall-open is detectable — [executor.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/hooks/executor.py).
- **OpenHands security analyzer:** an exception during analysis defaults to `HIGH` — [analyzer.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/security/analyzer.py); the ensemble does the same per child — [ensemble.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/security/ensemble.py).
- **OpenHands condenser:** a failed `SOFT` condensation returns the uncondensed view; a failed `HARD` one tries `hard_context_reset`, then re-raises — [condenser/base.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/context/condenser/base.py).

### Inferences
- A useful split for a kernel: observers fail open, enforcers fail closed, and the type of the hook (not its author's try/catch) decides. MAF arrived at this through two parallel mechanisms; OpenHands has the divide accidentally, between hooks and analyzer.
- SK's catch means an exception thrown by a filter before `next` blocks that one call but does not stop the loop, and the exception text goes to the model. This is read from code structure, not tested.
- OpenHands' `success=False` allow is a good audit pattern even if the default is debatable: it keeps "could not decide" distinct from "allowed".

### Gaps
- Mastra's behaviour for a generic exception in `processInputStep` and `processLLMRequest` was not traced individually; only the output and stream paths were read.
- AI SDK `resolve-tool-approval.ts` shows no try/catch when grepped, implying a throwing approval function propagates. I did not test it and found no doc statement.
- SK `KernelFunction.InvokeAsync` routes exceptions to `HandleException`; I did not read the rethrow path.

## Durability: event logs, checkpoints, and how resume interacts with hooks

### Takeaway
OpenHands is event-sourced with an append-only log, and condensation is itself an event over a derived view. Mastra and AI SDK's `WorkflowAgent` persist snapshots or steps and warn that replay can repeat side effects. Plain `ToolLoopAgent` and MAF agents leave persistence to the caller's message or session store. Across all of them, resume is under-specified for hooks: only MAF gates persistence on verdicts.

### Cited Findings

**OpenHands: event log and condensation**
- The SDK "defines an event-sourced state model with deterministic replay", with conversation state as the single source of truth — [SDK paper v2](https://arxiv.org/html/2511.03690v2).
- Condensation results are stored in the event log, letting the SDK preserve the whole log while keeping condensers stateless — [SDK paper v2 §4.6](https://arxiv.org/html/2511.03690v2).
- The log is append-only; `Condensation` events act like tombstones, and `View` derives what the LLM sees — [condenser README](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/context/condenser/README.md).
- `Condensation` carries `forgotten_event_ids`, optional `summary`, `summary_offset`, `llm_response_id`. `apply()` filters forgotten events and inserts a generated summary event at the offset — [event/condenser.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/event/condenser.py).
- `View.manipulation_indices` is the intersection of each property's safe boundaries. `enforce_properties` is a logged fallback that drops events until stable — [view.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/context/view/view.py).
- Four properties preserve pairing. `ToolCallMatchingProperty`: exactly one observation-like event per action `tool_call_id` — [tool_call_matching.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/context/view/properties/tool_call_matching.py).
- `BatchAtomicityProperty`: all events sharing `llm_response_id` are kept or forgotten together — [batch_atomicity.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/context/view/properties/batch_atomicity.py).
- `ToolLoopAtomicityProperty`: whole action/observation loops stay together, for Anthropic thinking blocks — [tool_loop_atomicity.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/context/view/properties/tool_loop_atomicity.py).
- `ObservationUniquenessProperty`: at most one observation per call, covering a crash-recovery synthetic error followed by a late real observation — [observation_uniqueness.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/context/view/properties/observation_uniqueness.py).
- `LLMSummarizingCondenser` defaults: `max_size=240`, `keep_first=2`, `minimum_progress=0.1`, `hard_context_reset_max_retries=5`, scaling 0.8 — [llm_summarizing_condenser.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/context/condenser/llm_summarizing_condenser.py).
- After an interrupt, actions with no observation are backfilled with a synthetic `AgentErrorEvent`, since providers reject orphaned tool calls — [local_conversation.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/conversation/impl/local_conversation.py).
- Appends are guarded by a lock file with a timeout — [event_store.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/conversation/event_store.py).
- Hook runs are logged as `HookExecutionEvent`, with output truncated to 50,000 characters — [conversation_hooks.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/hooks/conversation_hooks.py).
- `WAITING_FOR_CONFIRMATION` breaks the run loop. The next `run()` sets `RUNNING` with the code comment "(user approved)"; `reject_pending_actions` is the explicit refusal — [local_conversation.py](https://github.com/OpenHands/software-agent-sdk/blob/v1.60.2/openhands-sdk/openhands/sdk/conversation/impl/local_conversation.py).

**Microsoft Agent Framework**
- Workflow checkpoints are taken at the end of each superstep and capture executor state, pending messages, pending requests and responses, and shared state. Since Python 1.13.0, entry checkpoints are added before the first superstep and when responses are delivered — [Checkpoints](https://learn.microsoft.com/en-us/agent-framework/workflows/checkpoints).
- Executor hooks: `on_checkpoint_save` / `on_checkpoint_restore` (.NET `OnCheckpointingAsync` / `OnCheckpointRestoredAsync`). Resume requires the same topology and executor identities — [Checkpoints](https://learn.microsoft.com/en-us/agent-framework/workflows/checkpoints).
- Agent Hooks gates persistence: default after-run persistence waits for `output`. With `require_per_service_call_history_persistence=True`, each exchange persists after `post_model_call`, and a later `output` deny does not roll it back — [Agent Hooks](https://learn.microsoft.com/en-us/agent-framework/agents/agent-hooks).
- Agent Hooks buffers streaming until `post_model_call` and `output` pass — [Agent Hooks](https://learn.microsoft.com/en-us/agent-framework/agents/agent-hooks).
- Approval: a run ends with a request, then a new run in the same session carries the response. Responses must match a pending request in that `AgentSession` unless `disable_approval_response_binding` is set — [Tool approval](https://learn.microsoft.com/en-us/agent-framework/agents/tools/tool-approval).
- Approval is created before function middleware runs. A `pre_tool_call` transform of approval-bound arguments executes nothing under the original grant and returns a replacement request — [Agent Hooks](https://learn.microsoft.com/en-us/agent-framework/agents/agent-hooks).
- Standing approvals persist in `AgentSession` state and are not bound to the tool instance, so a replacement tool with the same name can match — [Tool approval](https://learn.microsoft.com/en-us/agent-framework/agents/tools/tool-approval).
- `CompactionProvider` on the chat client compacts only in-flight messages and leaves stored history unchanged; `IChatReducer` reduces the stored history. Tool calls and results form atomic groups — [MAF compaction](https://learn.microsoft.com/en-us/agent-framework/agents/conversations/compaction).

**Vercel AI SDK**
- `ToolLoopAgent` runs in memory and loses progress on a crash. In `WorkflowAgent`, tools marked `'use step'` are durable, retried 3 times by default, and completed model and tool steps are reused on replay — [WorkflowAgent](https://ai-sdk.dev/docs/agents/workflow-agent).
- Under `WorkflowAgent`, input callbacks replay after the model step finishes; `runtimeContext` and `toolsContext` must be serializable — [WorkflowAgent](https://ai-sdk.dev/docs/agents/workflow-agent).
- Approval does not pause `generate`. It returns `tool-approval-request` parts and the caller appends `tool-approval-response` and calls again — [Tool calling](https://ai-sdk.dev/docs/ai-sdk-core/tools-and-tool-calling).
- `experimental_toolApprovalSecret` HMAC-signs approvals so modified history cannot run tools — [WorkflowAgent](https://ai-sdk.dev/docs/agents/workflow-agent).

**Mastra**
- Workflow suspension writes a snapshot to storage, which survives deploys and restarts — [Suspend and resume](https://mastra.ai/docs/workflows/suspend-and-resume).
- Agent-run snapshots are deleted when the run finishes; the default in-memory store loses suspended runs on exit; `listSuspendedRuns` finds pending runs — [Agent approval](https://mastra.ai/docs/agents/agent-approval).
- Durable agents are beta. `recover` re-runs the loop from the last persisted snapshot and can reissue LLM calls and re-execute tools. A sub-agent delegation restarts in full. Per-call delegation hooks are not restored; use `defaultOptions.delegation` — [Durable agents](https://mastra.ai/docs/agents/durable-agents).
- Processor `state` is per request, created new for each `generate()` or `stream()` — [Processors](https://mastra.ai/docs/agents/processors).
- Observational Memory removes observed messages from context; unless `retrieval` is enabled, original wording is lost. It can block synchronously at `blockAfter` — [Observational Memory](https://mastra.ai/docs/memory/observational-memory).

### Inferences
- OpenHands' separation (immutable log, derived view, tombstone with event IDs) gives compaction audit and replay without a transcript rewrite. Its pairing guarantee is structural: compaction may cut only at boundaries valid for all properties, with enforcement as a backstop.
- MAF's `CompactionProvider` versus `IChatReducer` is the same in-flight versus durable distinction expressed as two registration points, and the docs warn the wrong one can persist synthetic summaries.
- Hook-closure state is the common resume hazard: Mastra loses per-call hooks, `WorkflowAgent` requires serializable context, and a Python closure or Rust trait object cannot be restored from a log. Durable decisions need to be data; hook code has to be re-resolved at load.
- Agent Hooks' default of one session per run suggests an approval pause and resume spans two hook sessions unless the application owns an emitter. This is my reading of the session description, not a documented statement about approvals.

### Gaps
- I did not confirm whether OpenHands re-runs `SessionStart` on resume. The code calls it during plugin load; issue [#5628](https://github.com/OpenHands/software-agent-sdk/issues/5628) implies hooks are resolved at load rather than restored.
- Exactly-once semantics for an executed-but-unrecorded tool were not found documented for OpenHands or MAF.
- How Mastra re-resolves processors after restart is not on the durable agents page; [#29137](https://github.com/mastra-ai/mastra/issues/29137) describes it from code.

## What problems have users reported?

### Takeaway
The recurring classes are enforcement that fails open, resume or replay repeating or skipping work, compaction breaking tool pairing, hooks that do not reach subagents, and missing tool-call identity at the hook boundary. Statements below are reporter claims from issue text. Open or closed state is as of 2026-11-08.

### Cited Findings

**OpenHands SDK**
- PreToolUse: a Claude Code style `hookSpecificOutput` deny is unparsed, and a crashed or timed-out policy hook lets the tool run — [#6120](https://github.com/OpenHands/software-agent-sdk/issues/6120) (open).
- Exit 1 with JSON `deny` is treated as blocking, contrary to documented semantics — [#6116](https://github.com/OpenHands/software-agent-sdk/issues/6116) (open).
- Agent-server is fail-open on corrupt `hooks.json` — [#6376](https://github.com/OpenHands/software-agent-sdk/issues/6376) — and on corrupt `settings.json`, where confirmation mode becomes `NeverConfirm` — [#6370](https://github.com/OpenHands/software-agent-sdk/issues/6370) (both open).
- Any user message implicitly approves pending actions on the next `run()` — [#5092](https://github.com/OpenHands/software-agent-sdk/issues/5092) (open). The "(user approved)" comment is confirmed in source.
- A plain message silently resumes a paused conversation — [#6010](https://github.com/OpenHands/software-agent-sdk/issues/6010) (open).
- A tool call discarded after `finish` runs on the next turn — [#6289](https://github.com/OpenHands/software-agent-sdk/issues/6289) (open).
- `_is_stuck_context_window_error` is a stub, so condensation repeats until the iteration limit — [#6104](https://github.com/OpenHands/software-agent-sdk/issues/6104) (open).
- The incremental view path skips `enforce_properties` after condensation, leaving orphaned tool results — [#4498](https://github.com/OpenHands/software-agent-sdk/issues/4498) (open).
- A poisoned `tool_result` causing `LLMBadRequestError` wedges a conversation with no recovery branch — [#5225](https://github.com/OpenHands/software-agent-sdk/issues/5225) (open).
- When summarization is unavailable the conversation fails instead of falling back to truncation — [#6048](https://github.com/OpenHands/software-agent-sdk/issues/6048) (open).
- Request for tool call id in hook input — [#6214](https://github.com/OpenHands/software-agent-sdk/issues/6214) (open). Agent hook evaluator runs ambient plugin hooks — [#6013](https://github.com/OpenHands/software-agent-sdk/issues/6013) (open). Duplicate plugin load runs hooks twice — [#5810](https://github.com/OpenHands/software-agent-sdk/issues/5810) (open).

**Vercel AI SDK**
- Approved calls run before the first step, so `onStepEnd` checkpoints miss their results. A crash then re-executes the approved side effect without asking; `abortSignal` is also not checked before that execution — [#24624](https://github.com/vercel/ai/issues/24624) (open, reporter on `ai` 7.0.162).
- `prepareStep` `activeTools` changes bust the provider prompt cache — [#14170](https://github.com/vercel/ai/issues/14170) (open).
- Requests for consuming pending user input before completing — [#22262](https://github.com/vercel/ai/issues/22262) — and steering across core functions — [#23640](https://github.com/vercel/ai/issues/23640) (both open).
- A provider-neutral per-step tool-call limit is requested — [#22920](https://github.com/vercel/ai/issues/22920) (open). `stopWhen` has no effect without tool calls — [#8544](https://github.com/vercel/ai/issues/8544) (open).
- `transformParams` not affecting `streamText` provider options — [#15704](https://github.com/vercel/ai/issues/15704) (closed).
- Subagent tools cannot use approval flows, a documented limit — [Subagents](https://ai-sdk.dev/docs/agents/subagents); nested approvals under code mode are requested — [#18143](https://github.com/vercel/ai/issues/18143) (open).

**Mastra**
- `DurableAgent.recover()` swallows processor-resolution errors and continues with no processors — [#29137](https://github.com/mastra-ai/mastra/issues/29137) (open).
- `TokenLimiter` removes the current run's tool call and result, so the agent loops and answers without the data — [#24110](https://github.com/mastra-ai/mastra/issues/24110) (open).
- `processOutputResult` tripwires were reported inconsistently between plain and durable agents — [#27170](https://github.com/mastra-ai/mastra/issues/27170) (closed); default abort reason differed on durable engines — [#27538](https://github.com/mastra-ai/mastra/issues/27538) (closed).
- An aborted approval can resume on another runtime if snapshot cleanup fails — [#29267](https://github.com/mastra-ai/mastra/issues/29267) (open). A stale suspended approval resumes without the user on every message — [#28342](https://github.com/mastra-ai/mastra/issues/28342) (open).
- Error processors cannot advance the model fallback chain — [#21280](https://github.com/mastra-ai/mastra/issues/21280) (open). In-band 200-status stream failures never trigger fallback, per the same issue.
- Pre-prompt ephemeral processor hook requested — [#25911](https://github.com/mastra-ai/mastra/issues/25911) (open). Snapshot before inference costs about 300 ms per turn, per reporter — [#24628](https://github.com/mastra-ai/mastra/issues/24628) (open).

**Microsoft Agent Framework**
- The tool-seam halt keys on the `host_error:` reason prefix, catching composition-produced denies — [#7561](https://github.com/microsoft/agent-framework/issues/7561) (open).
- No signal when an `as_tool()` child has no hooks bundle — [#8050](https://github.com/microsoft/agent-framework/issues/8050) (open). .NET function middleware for nested tools, for HITL — [#4158](https://github.com/microsoft/agent-framework/issues/4158) (open).
- Tool call id not exposed on `FunctionInvocationContext` — [#9454](https://github.com/microsoft/agent-framework/issues/9454) (open). Per-tool deadlines with cancellation-safe middleware requested — [#9235](https://github.com/microsoft/agent-framework/issues/9235) (open).
- Buffered streaming cost: requests for bounded-exposure incremental mode — [#7717](https://github.com/microsoft/agent-framework/issues/7717) — and skipping buffering when nothing enforces output — [#7716](https://github.com/microsoft/agent-framework/issues/7716) (both open).
- .NET middleware receives tool results as `JsonElement`, undocumented — [#9918](https://github.com/microsoft/agent-framework/issues/9918) (open). Checkpoint cadence configurability — [#9046](https://github.com/microsoft/agent-framework/issues/9046) (open).

**Semantic Kernel (superseded)**
- Python auto invocation lacks a runtime approval mechanism, reported as an indirect prompt-injection risk — [#14072](https://github.com/microsoft/semantic-kernel/issues/14072) (open).
- `extract_range()` drops a multi-call assistant message when filtering one parallel result — [#14485](https://github.com/microsoft/semantic-kernel/issues/14485) (open) — and reorders around call/result pairs — [#14486](https://github.com/microsoft/semantic-kernel/issues/14486) (open).
- Gemini connector did not invoke `IAutoFunctionInvocationFilter` — [#12998](https://github.com/microsoft/semantic-kernel/issues/12998) (closed).

### Inferences
- Tool-call identity keeps being requested at the hook boundary in OpenHands and MAF. A kernel should pass the call id to every tool-phase hook from the start.
- Compaction and pairing bugs recur in every runtime that edits a message list: SK's reducer, Mastra's `TokenLimiter`, OpenHands' incremental view. A central "safe cut point" invariant reduces this but, as #4498 shows, every code path must apply it.
- Approval resume bugs share a shape: the approval is treated as durable while the execution result is not. Checkpointing before executing an approved call does not fix it alone; an approved call needs a recorded intent and an idempotency or result receipt.

### Gaps
- Issue search was keyword-based and capped to recent relevant hits, so this is a sample, not a census. Frequency across runtimes cannot be compared from it.
- Most issue bodies were read only in part. I independently confirmed the #5092 code comment; the other reports are unverified.
