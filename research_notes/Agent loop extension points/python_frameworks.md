# Agent loop extension points: LangChain/LangGraph, Google ADK, Pydantic AI, OpenAI Agents SDK

Checked 2026-11-08. Versions seen on the GitHub release pages that day: langchain-core 1.7.6 and langgraph-sdk 0.4.14 (the release list did not show the `langchain` or `langgraph` package versions) ([LangChain releases](https://github.com/langchain-ai/langchain/releases), [LangGraph releases](https://github.com/langchain-ai/langgraph/releases)); ADK Python v2.14.0 ([releases](https://github.com/google/adk-python/releases)); Pydantic AI v2.66.0, with v1.107.10 still receiving maintenance releases ([releases](https://github.com/pydantic/pydantic-ai/releases)); OpenAI Agents SDK v0.24.0 stable, v0.24.0-alpha.7 pre-release ([releases](https://github.com/openai/openai-agents-python/releases)).

Moved docs: `google.github.io/adk-docs` now 301s to `adk.dev`; `ai.pydantic.dev` now 301s to `pydantic.dev/docs/ai`. Source links below point at each repo's default branch as read on the check date, so line content can drift.

## Which loop moments are hookable?

### Takeaway
Pydantic AI has the widest typed surface (before/after/wrap/error for run, node, model request, tool validation, tool execution, output validation and output processing). ADK and LangChain cover the main moments with smaller sets. The OpenAI SDK splits the job: `RunHooks`/`AgentHooks` are return-less lifecycle callbacks, and control lives in guardrails, approvals and `RunConfig` filters.

### Cited Findings

**LangChain 1.x middleware**
- Node-style hooks: `before_agent` and `after_agent` (once per invocation), `before_model` and `after_model` (around each model call). Wrap-style hooks: `wrap_model_call`, `wrap_tool_call`. Async variants exist. — [Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom)
- No dedicated hook is listed for model error, tool failure or output validation. Those are handled inside `wrap_*`, as the built-in retry middleware do. — [Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom); [Built-in middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in)
- Limits, retries, fallback and history are shipped as middleware: `ModelCallLimit`, `ToolCallLimit`, `ModelRetry`, `ToolRetry`, `ModelFallback`, `Summarization`, `ContextEditing`, `PII`, `LLMToolSelector`, `LLMToolEmulator`, `HumanInTheLoop`. — [Built-in middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in)
- Model, tools and each middleware node are added as graph nodes in `create_agent`. — [factory.py](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py)
- HITL pauses before tool calls, once per model turn by default, or per tool call with `interrupt_mode="per_call"`. It requires a checkpointer. — [Built-in middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in)
- Decision types are `approve`, `edit`, `reject`, `respond`. — [human_in_the_loop.py](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/human_in_the_loop.py)

**LangGraph**
- The general pause primitive is `interrupt()` inside a node, resumed with `Command(resume=...)`. Payloads surface in `result["__interrupt__"]` or `stream.interrupts`. — [Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)

**Google ADK**
- Plugin callbacks: `on_user_message_callback`, `before_run_callback`, `before_agent`/`after_agent`, `before_model`/`after_model`, `on_model_error_callback`, `before_tool`/`after_tool`, `on_tool_error_callback`, `on_event_callback`, `after_run_callback`, plus Python's `on_agent_error_callback`, `on_run_error_callback` and `close()`. — [Plugins](https://adk.dev/plugins/); [base_plugin.py](https://github.com/google/adk-python/blob/main/src/google/adk/plugins/base_plugin.py)
- Object-level callbacks: `before_agent`, `after_agent`, `before_model`, `after_model`, `before_tool`, `after_tool`. — [Callbacks](https://adk.dev/callbacks/)
- `ReflectAndRetryToolPlugin` retries tool failures with a configurable `max_retries`; a companion model plugin asks the model to retry after a response that ends in an error such as a malformed function call. — [reflect_retry_tool_plugin.py](https://github.com/google/adk-python/blob/main/src/google/adk/plugins/reflect_retry_tool_plugin.py); [Plugins](https://adk.dev/plugins/)
- Compaction is configuration, not a callback: `EventsCompactionConfig` with `compaction_interval`, `overlap_size`, `token_threshold`, `event_retention_size`, `summarizer`; token-based takes priority over sliding window. — [Compaction](https://adk.dev/context/compaction/)
- HITL: `FunctionTool(..., require_confirmation=True)` or `request_confirmation` (experimental). — [Tool confirmation](https://adk.dev/tools-custom/confirmation/)
- ADK also ships `ToolCallIntegrityPlugin`, which HMAC-stamps stored function calls and lets a tool run only with a valid stamp, because stored calls can re-run on resume and a session-store writer could change them. — [_tool_call_integrity_plugin.py](https://github.com/google/adk-python/blob/main/src/google/adk/plugins/_tool_call_integrity_plugin.py)

**Pydantic AI**
- `Hooks` families, each with before/after, `wrap`, and error: run, node, model request, tool validation, tool execution, output validation, output processing. Also `prepare_model_request` (per attempt), `prepare_tools`, `prepare_output_tools`, `deferred_tool_calls`, `run_event_stream`, `event`. — [Hooks](https://pydantic.dev/docs/ai/core-concepts/hooks/); [capabilities/abstract.py](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/capabilities/abstract.py)
- A capability bundles tools, hooks, instructions, model settings and model selection. Capabilities can also publish typed events; a `dispatch='immediate'` listener can veto an action before it proceeds. — [Capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/)
- `ProcessHistory` wraps a history processor and is a thin layer over `before_model_request`. It runs before every model request in a run, not only the first. — [Message history](https://pydantic.dev/docs/ai/core-concepts/message-history/)
- Compaction: a provider-native `Compaction` capability and a model-agnostic Harness Compaction. — [Capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/)
- `@agent.output_validator` takes `RunContext` and the output and returns the output. — [Output](https://pydantic.dev/docs/ai/core-concepts/output/)
- `UsageLimits`: `request_limit` (default 50), `tool_calls_limit`, `input_tokens_limit`, `output_tokens_limit`, `per_request_input_tokens_limit`, `cost_limit`, `count_tokens_before_request`. Tools and capabilities can read limits from `ctx.usage_limits`. — [Agent](https://pydantic.dev/docs/ai/core-concepts/agent/)
- HITL: `requires_approval=True`, `ApprovalRequired`, `CallDeferred`; the run ends with `DeferredToolRequests` and resumes with `deferred_tool_results`. — [Deferred tools](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/)

**OpenAI Agents SDK**
- `RunHooks`: `on_llm_start`, `on_llm_end`, `on_agent_start`, `on_agent_end`, `on_handoff`, `on_tool_start`, `on_tool_end`. `AgentHooks` has per-agent equivalents (`on_start`, `on_end`, `on_handoff`, `on_tool_start`, `on_tool_end`, `on_llm_start`, `on_llm_end`). `on_tool_*` is documented for local tools. — [Lifecycle reference](https://openai.github.io/openai-agents-python/ref/lifecycle/)
- Guardrails: input (first agent only), output (last agent only), and tool input/output guardrails on `FunctionTool` for every invocation. — [Guardrails](https://openai.github.io/openai-agents-python/guardrails/)
- `RunConfig` controls: `call_model_input_filter` (edits instructions and input right before the model call), `session_input_callback`, `handoff_input_filter`, `tool_error_formatter`. `error_handlers` accepts `max_turns`, `model_refusal`, `invalid_final_output`. — [Running agents](https://openai.github.io/openai-agents-python/running_agents/)
- Compaction: `OpenAIResponsesCompactionSession`, automatic after each turn when `should_trigger_compaction` is true, or manual with `run_compaction()`. — [Sessions](https://openai.github.io/openai-agents-python/sessions/)
- Tool approval: `needs_approval` on function tools, `Agent.as_tool()`, `ShellTool`, `ApplyPatchTool`; `require_approval` on local MCP; `tool_config` on hosted MCP. — [Human in the loop](https://openai.github.io/openai-agents-python/human_in_the_loop/)
- Tool guardrails do not cover the handoff call itself, hosted tools, or built-in execution tools such as `ComputerTool`, `ShellTool`, `ApplyPatchTool`. `Agent.as_tool()` exposes no tool guardrail options. Local MCP servers can set them. — [Guardrails](https://openai.github.io/openai-agents-python/guardrails/)

### Inferences
Coverage summary, derived from the findings above ("config" means a setting or built-in, not a general hook):

| Moment | LangChain | ADK | Pydantic AI | OpenAI |
|---|---|---|---|---|
| Run start/end | `before/after_agent` | `before/after_run`, `on_user_message` | `before/after/wrap_run` | `on_agent_start/end` (per agent change, no run-level pair) |
| Before/after model | yes | yes | yes | observe only, plus `call_model_input_filter` |
| Wrap model | `wrap_model_call` | none | `wrap_model_request` | none found |
| Model error/retry | inside wrap | `on_model_error` | `on_model_request_error`, `ModelRetry` | `ModelSettings.retry` config, `error_handlers` for three cases |
| Before/after tool | via wrap | yes | validate and execute, separately | observe only, plus tool guardrails |
| Tool failure | inside wrap | `on_tool_error` | `on_tool_execute_error`, `on_tool_validate_error` | `tool_error_formatter` |
| Output validation | none named (`after_model`, structured-output handling) | `after_model` | validators and output hooks | output guardrails |
| Limits | middleware | not confirmed | `UsageLimits` config | `max_turns` + handler |
| History edits | middleware | config | `ProcessHistory`, `before_model_request` | input filter, session callback, session wrapper |
| Handoff/subagent | none named | `before/after_agent` | none named | `on_handoff`, handoff filter |
| HITL | `interrupt()` | confirmation | deferred tools | `needs_approval` |

### Gaps
- ADK run limits (for example an LLM-call cap) were not checked in a page or source read here.
- No dedicated subagent hook was found for LangChain or Pydantic AI in the pages read. Absence from those pages is not proof of absence.
- Which node-style hook `SummarizationMiddleware` and `ContextEditing` use was not confirmed.

## What may a hook do at each moment?

### Takeaway
All frameworks except OpenAI's lifecycle hooks can short-circuit a model or tool call. Retry is first-class only in LangChain's wrap hooks and Pydantic AI's `ModelRetry`; ADK gets it through error callbacks and retry plugins. OpenAI lifecycle hooks return nothing, so blocking belongs to guardrails and approval.

### Cited Findings

**LangChain**
- Node hooks return a state update; `before_model` and `after_model` can also `jump_to` `'end'`, `'tools'` or `'model'` when declared in `can_jump_to`. — [Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom)
- `wrap_model_call` and `wrap_tool_call` may call the handler zero times (short-circuit), once, or several times (retry). `wrap_model_call` can return `ExtendedModelResponse` with a `Command`; `wrap_tool_call` returns a `Command` directly. — [Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom)
- If an outer middleware retries the handler, `Command`s from earlier attempts are discarded. — [Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom)
- `ModelCallLimitMiddleware` stops by returning `{"jump_to": "end", "messages": [...]}` or raises `ModelCallLimitExceededError`, per `exit_behavior`. — [model_call_limit.py](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/model_call_limit.py)
- `ToolCallLimit` defaults to `exit_behavior='continue'`, which blocks excess calls with errors. — [Built-in middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in)
- `ModelRetry` and `ToolRetry` default to `max_retries=2` and `on_failure='continue'`. — [Built-in middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in)

**ADK**
- `before_model` returning `LlmResponse` skips the LLM; `after_model` returning one replaces the response. `before_tool` returning a dict skips the tool and is used as its result; `after_tool` returning a dict replaces it. `before_agent` returning `Content` skips the agent; `after_agent` returning `Content` appends an event rather than replacing output. — [Callbacks](https://adk.dev/callbacks/)
- Plugin-only: `on_user_message_callback` can replace the message; `before_run_callback` returning `Content` halts the run; `on_event_callback` can override an event, but `id`, `invocation_id` and `timestamp` stay from the original in Python. — [Plugins](https://adk.dev/plugins/)
- `on_model_error_callback` returning `LlmResponse`, or `on_tool_error_callback` returning a dict, suppresses the exception and `after_*` then runs normally. Returning `None` re-raises. — [Plugins](https://adk.dev/plugins/)
- `after_run_callback`, `on_agent_error_callback` and `on_run_error_callback` cannot change the outcome. — [Plugins](https://adk.dev/plugins/)

**Pydantic AI**
- `before_*` and `after_*` can return modified values. `before_model_request` can reassign `request_context.messages` to affect only the current request; appending in place is deprecated. — [Hooks](https://pydantic.dev/docs/ai/core-concepts/hooks/)
- Skip by raising `SkipModelRequest(response)`, `SkipToolValidation(args)` or `SkipToolExecution(result)`. A wrap hook that does not call its handler skips everything inside. — [Hooks](https://pydantic.dev/docs/ai/core-concepts/hooks/)
- `ModelRetry` retries: model hooks send `RetryFeedbackPart` and count against the output budget; tool hooks count against that tool's `max_retries`. `ToolFailed` reports a definitive failure without using retry budget. — [Hooks](https://pydantic.dev/docs/ai/core-concepts/hooks/)
- `ApprovalRequired` or `CallDeferred` may be raised from `after_tool_validate` or `before_tool_execute`; from `before_tool_validate` or `tool_validate_error` it is a `UserError`. — [Hooks](https://pydantic.dev/docs/ai/core-concepts/hooks/)
- Error hooks can re-raise (the default), raise something else, or return a result to recover. — [Hooks](https://pydantic.dev/docs/ai/core-concepts/hooks/)
- `prepare_tools` filtering hides tools and also blocks their execution. — [Hooks](https://pydantic.dev/docs/ai/core-concepts/hooks/)
- Retry budget defaults to 1, configured by `retries`/`AgentRetries`, per tool, or per output. Tool retries are per tool; text output retries share one budget per run and end in `UnexpectedModelBehavior`. — [Agent](https://pydantic.dev/docs/ai/core-concepts/agent/)
- Output validators can replace the output through their return value and reject with `ModelRetry`. `ToolFailed` is not supported in them. They also run on each partial output while streaming. — [Output](https://pydantic.dev/docs/ai/core-concepts/output/)
- Returning a replacement event from the `event` hook is deprecated. — [Hooks](https://pydantic.dev/docs/ai/core-concepts/hooks/)

**OpenAI**
- No `RunHooks`/`AgentHooks` method returns a value, and the reference documents no way to cancel or veto. The hooks do receive mutable objects such as `input_items`, but the page does not say whether changing them affects the request. — [Lifecycle reference](https://openai.github.io/openai-agents-python/ref/lifecycle/)
- `on_llm_start` receives `filtered.input` after `call_model_input_filter`, and `mark_input_as_sent(filtered.input)` runs before the hooks. — [run_loop.py](https://github.com/openai/openai-agents-python/blob/main/src/agents/run_internal/run_loop.py)
- Input guardrails trip with `InputGuardrailTripwireTriggered`; `run_in_parallel=False` guarantees the model does not start. Output guardrails only reject. Input tool guardrails can skip the call, replace output, or trip; output tool guardrails can replace output or trip. — [Guardrails](https://openai.github.io/openai-agents-python/guardrails/)
- Tool input guardrails run before `on_tool_start` in the function-tool path. — [tool_execution.py](https://github.com/openai/openai-agents-python/blob/main/src/agents/run_internal/tool_execution.py)
- `error_handlers` return a controlled final output instead of the error. — [Running agents](https://openai.github.io/openai-agents-python/running_agents/)

### Inferences
- Pydantic AI's exception-based skip and LangChain's "don't call handler" are equivalent in power; ADK uses "non-`None` return means override"; OpenAI concentrates authority in guardrails.
- A closed set of hook outcomes is implicit in Pydantic AI (`Skip*`, `ModelRetry`, `ToolFailed`, `ApprovalRequired`) and is the nearest analogue to a typed decision enum in a Rust kernel.

### Gaps
- Whether mutating `input_items` in `on_llm_start` changes the actual request was not established; `call_model_input_filter` is the documented route.

## How do multiple hooks compose?

### Takeaway
LangChain and Pydantic AI use an onion for wraps with reverse-order after hooks. ADK runs a flat chain where the first non-`None` return wins and plugins precede object callbacks. OpenAI runs the run-level and agent-level hook concurrently.

### Cited Findings
- LangChain: `before_*` first to last; `wrap_*` nested with the first middleware outermost; `after_*` last to first. — [Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom)
- LangChain: nested `Command`s apply inner first then outer, so the outermost wins on non-reducer fields; reducer fields such as messages are additive. — [Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom)
- ADK: plugins run in list order; the first non-`None` result stops the chain and skips the matching object-level callback. — [Plugins](https://adk.dev/plugins/); [plugin_manager.py](https://github.com/google/adk-python/blob/main/src/google/adk/plugins/plugin_manager.py)
- ADK tool path order is plugin `before_tool`, then agent `before_tool` if the plugin returned `None`, then the tool, then plugin `after_tool`, then agent `after_tool` if the plugin returned `None`. — [_caller.py](https://github.com/google/adk-python/blob/main/src/google/adk/flows/llm_flows/tools/_caller.py)
- ADK error observers are different: `on_agent_error_callback` and `on_run_error_callback` run on every plugin. — [Plugins](https://adk.dev/plugins/)
- Pydantic AI, across capabilities: `before_*` in capability order; `after_*` and `on_*_error` in reverse; `wrap_*` nested with first outermost; all wrappers are entered before any `before_*` runs. Inside one `Hooks` instance, before/after/error fire in registration order and wraps nest. — [Hooks](https://pydantic.dev/docs/ai/core-concepts/hooks/)
- Pydantic AI docs advise putting mandatory policy checks in an outer wrapper, because an outer short-circuit skips everything inside. — [Hooks](https://pydantic.dev/docs/ai/core-concepts/hooks/)
- Pydantic AI capabilities can declare `get_ordering()` with `outermost`/`innermost`, `wraps`/`wrapped_by` and `requires`. A topological sort keeps list order as tie-break and raises `UserError` on missing requirements or cycles. — [abstract.py](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/capabilities/abstract.py); [_ordering.py](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/capabilities/_ordering.py)
- Pydantic AI per-attempt versus per-turn: `before_model_request` and `wrap_model_request` run per model turn; `prepare_model_request`, `on_model_request_error` and `after_model_request` run per attempt. — [Hooks](https://pydantic.dev/docs/ai/core-concepts/hooks/)
- An open issue is titled "V3: simplify capability composition and lifecycle dispatch". Title only was read. — [pydantic-ai #7339](https://github.com/pydantic/pydantic-ai/issues/7339)
- OpenAI: run-level and agent-level hooks are awaited together through `gather_with_cancel`, for example `on_llm_start` and `on_tool_start`. — [run_loop.py](https://github.com/openai/openai-agents-python/blob/main/src/agents/run_internal/run_loop.py); [tool_execution.py](https://github.com/openai/openai-agents-python/blob/main/src/agents/run_internal/tool_execution.py)
- OpenAI guardrails are started as tasks and consumed with `asyncio.as_completed`; the first tripwire cancels the rest. — [guardrails.py](https://github.com/openai/openai-agents-python/blob/main/src/agents/run_internal/guardrails.py)

### Inferences
- Only Pydantic AI offers a declared ordering constraint. LangChain and ADK depend on list position, so a security hook's outermost status is a convention.
- ADK's first-non-`None` rule means a plugin can pre-empt a later plugin's check. In the before-tool path, an earlier plugin returning a dict prevents a later plugin's veto.
- OpenAI's concurrent start hooks have no defined relative order.

### Gaps
- ADK's semantics for lists of object-level callbacks are not on the Callbacks page. The name `canonical_*_callbacks` in source suggests lists exist; behaviour was not verified.
- The order of multiple Pydantic AI output validators is not on the Output page.

## What happens when a hook raises?

### Takeaway
None of these documents a fail-open mode for an ordinary pre-call hook. Exceptions generally end the run; ADK wraps them in `RuntimeError`, Pydantic AI supports per-hook timeouts, and OpenAI documents fail-closed for approval callables. Control-flow exceptions need to pass through hook-level handlers untouched.

### Cited Findings
- LangChain docs do not state the behaviour of a raising hook; they only advise handling errors gracefully, and the retry example re-raises on the last attempt. — [Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom)
- LangChain built-in retry and error middleware explicitly re-raise `GraphBubbleUp` so interrupts and parent commands are not treated as failures. — [model_retry.py](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/model_retry.py); [tool_error.py](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/tool_error.py)
- This was a real defect: `ModelRetryMiddleware` and `ModelFallbackMiddleware` swallowed `GraphInterrupt`; closed 2026-08-02. — [langchain #38837](https://github.com/langchain-ai/langchain/issues/38837)
- LangGraph docs warn that `interrupt()` pauses by raising, so a bare `try/except` can catch it. — [Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- ADK: an exception in a plugin callback is logged and re-raised as `RuntimeError` with the original as `__cause__`; remaining plugins do not run. — [plugin_manager.py](https://github.com/google/adk-python/blob/main/src/google/adk/plugins/plugin_manager.py); [Plugins](https://adk.dev/plugins/)
- ADK: exceptions in `on_agent_error_callback` and `on_run_error_callback` are logged, not raised, and all plugins still run. — [plugin_manager.py](https://github.com/google/adk-python/blob/main/src/google/adk/plugins/plugin_manager.py); [Plugins](https://adk.dev/plugins/)
- Pydantic AI: each hook accepts `timeout`; exceeding it raises `HookTimeoutError` with `hook_name` and `timeout`. — [Hooks](https://pydantic.dev/docs/ai/core-concepts/hooks/)
- Pydantic AI: error hooks cover failures from the core operation only. `ModelRetry` from a before/after/wrap hook bypasses that stage's error hook. — [Hooks](https://pydantic.dev/docs/ai/core-concepts/hooks/)
- Pydantic AI: a run that raises returns no result, and its messages are lost unless wrapped in `capture_run_messages`. — [Agent](https://pydantic.dev/docs/ai/core-concepts/agent/)
- OpenAI: `needs_approval` callables fail closed; missing, malformed or non-object arguments, or values like `NaN`, require manual approval. — [Human in the loop](https://openai.github.io/openai-agents-python/human_in_the_loop/)
- OpenAI: if an output guardrail function raises rather than returning a verdict, the SDK persists the final-turn items and surfaces the exception. — [Guardrails](https://openai.github.io/openai-agents-python/guardrails/)
- OpenAI: if a guardrail raises, sibling guardrail tasks are cancelled and awaited. — [guardrails.py](https://github.com/openai/openai-agents-python/blob/main/src/agents/run_internal/guardrails.py)

### Inferences
- With no catch documented around LangChain node hooks, a raising hook most likely fails the graph run, with the last checkpoint available for retry. This was not tested.
- OpenAI `on_llm_start` is awaited with no surrounding handler in the code excerpt read, so an exception there probably propagates. Not tested.
- For a kernel, ADK's split is a clean model: control hooks abort on error, observer hooks log and continue.

### Gaps
- OpenAI docs do not state what a raising `RunHooks`/`AgentHooks` method does, and whether `on_tool_start` failures become tool errors could not be determined from the excerpts read.
- ADK object-level callback exceptions are not documented on the Callbacks page.

## Durability: checkpointing and how resume interacts with hooks

### Takeaway
LangGraph and ADK re-enter code on resume, so anything before the pause runs again. Pydantic AI journals model and tool steps only under a durable engine, while hook code in orchestration can re-run. OpenAI serializes a `RunState` at approval pauses and warns there is no exactly-once. All four leave side-effect idempotency to the author.

### Cited Findings

**LangGraph and LangChain**
- Resume does not continue from the `interrupt()` line: the node restarts from the beginning, so earlier code runs again. Side effects before an interrupt should be idempotent or moved after it or into another node. — [Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- Multiple interrupts in one node match resume values strictly by index; do not conditionally skip them. Parallel interrupts can be resumed with a map of IDs to values. — [Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- Durability modes: `exit`, `async`, `sync`. Checkpoints occur at super-step boundaries; writes from successful sibling nodes are kept and not re-run. — [Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers.md)
- `HumanInTheLoopMiddleware` calls `interrupt()` inside the middleware. — [human_in_the_loop.py](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/human_in_the_loop.py)
- Thread-level limits in `ModelCallLimit` and `ToolCallLimit` need a checkpointer. — [Built-in middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in)

**ADK**
- Resumability uses `ResumabilityConfig(is_resumable=True)` on `App`. Resume is by original `invocation_id`, through `/run_sse` or `runner.run_async`; ADK Web and CLI resume are not supported. — [Resume](https://adk.dev/runtime/resume/)
- Tools run at least once and may run more than once on resume; completed tool results are reinstated and the failed tool re-runs. Custom agents need their own state. — [Resume](https://adk.dev/runtime/resume/)
- Tool confirmation does not support `DatabaseSessionService` or `VertexAiSessionService`, per the docs; the examples show the same tool function running again with confirmation data. — [Tool confirmation](https://adk.dev/tools-custom/confirmation/)

**Pydantic AI**
- Durable engines: Temporal, DBOS, Prefect, Restate, AWS Lambda, Absurd, pgtask, Azure Durable Task, Apache Airflow. Each journals model requests and wrapped tool calls; restart replays completed steps and retries the in-flight one. — [Durable execution](https://pydantic.dev/docs/ai/integrations/durable_execution/overview/)
- Custom hook work runs durably only if moved into engine activities, steps or tasks through durable capability operations. — [Durable execution](https://pydantic.dev/docs/ai/integrations/durable_execution/overview/)
- Without an engine, resume is by passing captured messages as `message_history`. Executed tool calls are not repeated, but usage and limits restart from zero. — [Agent](https://pydantic.dev/docs/ai/core-concepts/agent/)
- A deferred resume is a new run with a new `run_id` and the same `conversation_id`. Approval is not an authorization boundary against an untrusted client. — [Deferred tools](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/)

**OpenAI**
- Resume: `result.to_state()`, `state.approve()`/`state.reject()`, then `Runner.run(agent, state)` with the original top-level agent. `to_json`/`from_json` round-trip it; sticky decisions are stored in `RunState`. — [Human in the loop](https://openai.github.io/openai-agents-python/human_in_the_loop/)
- Not exactly-once: a crash after a side effect and before saving can repeat the action, so use idempotency keys. `from_json` does not verify the snapshot or the submitter. — [Human in the loop](https://openai.github.io/openai-agents-python/human_in_the_loop/)
- Tool input guardrails normally run after approval, immediately before execution. `ToolExecutionConfig(pre_approval_tool_input_guardrails=True)` also runs them before the pause, and calls are rechecked after approval. — [Guardrails](https://openai.github.io/openai-agents-python/guardrails/)
- Start hooks are not replayed before pending approval calls are resolved, so hook-created tools must be rebuilt on restore. — [Lifecycle reference](https://openai.github.io/openai-agents-python/ref/lifecycle/)

### Inferences
- Because HITL is an `interrupt()` in a graph node and nodes restart on resume, LangChain's HITL hook logic before the `interrupt()` call runs twice. It is safe only if that logic is pure.
- Pydantic AI's non-durable resume resets usage counters, which means a budget does not survive a resume unless the application carries it.
- Recheck-after-approval (OpenAI) is the pattern that closes the gap between what was approved and what is run.

### Gaps
- ADK Resume docs do not say how callbacks and plugins behave on resume. An open issue asks for a replay indicator, implying they re-fire for rerun nodes (see next section).
- Pydantic AI's engine pages were not read individually, so per-engine hook restrictions are not listed.
- LangGraph's Durable Execution page returned little on this; the Checkpointers page was used instead.

## What problems have users reported?

### Takeaway
The recurring failures are orphaned or dropped tool-call pairs after trimming or compaction, control-flow exceptions swallowed by retry hooks, HITL decisions not enforced in edge cases, resume re-running side effects, and guardrails that act after the effect. Many issues are recent; several are closed, and some of those fixes landed in the last few months.

### Cited Findings

**Orphaned tool calls and history edits**
- LangChain: `_find_safe_cutoff_point` reportedly preserves orphan `ToolMessage`s, leading to provider errors (open). — [langchain #38352](https://github.com/langchain-ai/langchain/issues/38352)
- LangChain: current source searches back for the matching `AIMessage` and otherwise advances past tool messages to avoid orphans. — [summarization.py](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/summarization.py)
- LangChain: summarization removing Anthropic thinking blocks (open; title read). — [langchain #34794](https://github.com/langchain-ai/langchain/issues/34794)
- ADK: compaction folded a long-running tool's call into a summary, then resume failed with "No function call event found" (closed 2026-07-08). — [adk-python #5602](https://github.com/google/adk-python/issues/5602)
- ADK: one orphaned `function_call` silently and permanently blocked all later compaction (closed 2026-10-02). — [adk-python #7236](https://github.com/google/adk-python/issues/7236)
- OpenAI: compaction at an approval interruption swallowed the pending `function_call`, leaving an orphan output after resume (closed 2026-10-13). — [openai-agents #5340](https://github.com/openai/openai-agents-python/issues/5340)
- OpenAI: output-guardrail trip on a streamed resume left an orphaned call; the report notes `drop_orphan_function_calls` strips it (closed 2026-08-02). — [openai-agents #4125](https://github.com/openai/openai-agents-python/issues/4125)
- OpenAI: a finished parallel tool call was dropped when another tool's guardrail tripped, so the next turn repeated the side effect (closed 2026-10-14). — [openai-agents #5327](https://github.com/openai/openai-agents-python/issues/5327)
- Pydantic AI addressed this structurally: before each model request it synthesizes an `interrupted` `ToolReturnPart` for a call with no result, drops orphaned results, and merges adjacent messages. Docs warn this silently changes what the model sees. — [Message history](https://pydantic.dev/docs/ai/core-concepts/message-history/)

**Unbounded or wrong retries**
- LangChain: `ToolStrategy` retries structured-output validation failures with no cap (open, with a proposed opt-in `max_retries`). — [langchain #40753](https://github.com/langchain-ai/langchain/issues/40753)
- LangChain: `retry_on=SomeError` as a bare class retried every exception (closed; title read). — [langchain #41025](https://github.com/langchain-ai/langchain/issues/41025)

**HITL and resume**
- LangChain: batched-mode HITL silently drops a reviewer's edit and runs the original for empty-string tool IDs (open). The same report cites a cross-tool edit bypassing `interrupt_on` (#40166) and `jump_to="tools"` bypassing the HITL gate (#40492). — [langchain #41522](https://github.com/langchain-ai/langchain/issues/41522)
- LangChain: approval UIs that reorder actions need stable decision IDs (open; title read). — [langchain #40206](https://github.com/langchain-ai/langchain/issues/40206)
- LangGraph: `interrupt()` in a loop causes extra resumes (open). — [langgraph #7780](https://github.com/langchain-ai/langgraph/issues/7780)
- LangGraph: long tool calls of about 180s silently re-executed from checkpoint on LangGraph Cloud, as reported (open). — [langgraph #7417](https://github.com/langchain-ai/langgraph/issues/7417)
- LangGraph: durability docstrings can be read as implying exactly-once for external effects (open). — [langgraph #9240](https://github.com/langchain-ai/langgraph/issues/9240)
- ADK: resume re-executes orchestrator node bodies, so logs, metrics and alert hooks fire again, and nothing in `Context` marks replay (open). — [adk-python #7943](https://github.com/google/adk-python/issues/7943)
- Pydantic AI: capability hooks run in re-executed orchestration code, so external writes can commit more than once. The report is self-described as filed by a Claude Code session, verified against 2.22.0 and temporalio 1.27.1. — [pydantic-ai #7176](https://github.com/pydantic/pydantic-ai/issues/7176)
- OpenAI: deferred interrupted-turn session items were never persisted on an approval resume path (closed; title read). — [openai-agents #4827](https://github.com/openai/openai-agents-python/issues/4827)

**Guardrails too late**
- OpenAI docs state that in parallel mode the model call may have happened and tools may have run before a tripwire; output guardrails and tool output guardrails run after the fact. — [Guardrails](https://openai.github.io/openai-agents-python/guardrails/)
- OpenAI: `Runner.run_streamed` sends a model request after a parallel input guardrail has already tripped (open). — [openai-agents #5469](https://github.com/openai/openai-agents-python/issues/5469)
- OpenAI docs: an output guardrail rejection keeps completed tool items in the session and replaces a rejected terminal function-tool output with a placeholder, "Output withheld by an output guardrail." — [Guardrails](https://openai.github.io/openai-agents-python/guardrails/)
- Community requests for a pre-execution gate and a budget guard in the OpenAI SDK exist as closed issues (titles read). — [openai-agents #2970](https://github.com/openai/openai-agents-python/issues/2970); [openai-agents #3353](https://github.com/openai/openai-agents-python/issues/3353)

### Inferences
- Tool-call pairing is the most repeated bug class across all four frameworks, and it appears at the interaction of compaction with pending or long-running calls. Pydantic AI's repair at the model boundary is the only central fix found; the others patch individual paths.
- Issue volume reflects project popularity and community filing habits, so it does not rank frameworks by quality.

### Gaps
- Several recent issues appear to be LLM-assisted or come from a small set of filers; only some were read in full, and none were reproduced here.
- For items marked "title read", only the title and status were checked.
- Searches for Pydantic AI orphan or infinite-retry bugs found little current material, which may reflect the search terms rather than an absence of reports.
