# Gemma Thinking Design

## Goal

Enable native thinking for the local Gemma 4 model served by llama.cpp while
preserving AIDaemon's current channel behavior: users receive only the final
answer, never the model's private reasoning text.

## Current State

The primary provider is an OpenAI-compatible llama.cpp server at
`http://127.0.0.1:8081/v1`, using `gemma-4-26b`. The loaded Gemma chat template
supports an `enable_thinking` template variable. A direct request confirmed
that llama.cpp returns the resulting trace in `message.reasoning_content`.

AIDaemon currently supports a generic `reasoning = { effort = ... }` request
field and parses `message.reasoning` and `message.reasoning_details`. It does
not send llama.cpp's Gemma-specific thinking fields and does not parse
`reasoning_content` on buffered responses.

## Configuration

Add an opt-in OpenAI-compatible provider setting for llama.cpp/Gemma thinking.
When enabled, requests include:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": true
  },
  "reasoning_format": "deepseek"
}
```

The setting belongs to the provider configuration rather than being inferred
from the model name. This avoids sending llama.cpp-specific fields to cloud
providers or other OpenAI-compatible servers.

The local primary provider enables this setting in `config.toml`. Its
`max_tokens` increases from `4096` to `8192`, giving Gemma room for both its
thinking trace and final answer. Fallback-provider behavior remains unchanged.

## Provider Behavior

The OpenAI-compatible provider adds the llama.cpp fields only when the new
setting is enabled. Existing generic reasoning-effort behavior remains
independent and unchanged.

Buffered response parsing accepts `message.reasoning_content` in addition to
the existing `reasoning` and `reasoning_details` response shapes. Streaming
already recognizes `reasoning_content` deltas and requires no behavioral
change.

Parsed reasoning is stored only in `ProviderResponse.thinking`. Final answer
text remains in `ProviderResponse.content`; the two values are never merged.

## Channel Behavior

Telegram, Slack, Discord, and other channels continue sending the sanitized
final `content` response exactly as before. They may show the existing generic
“Thinking...” activity indicator, but they do not display or persist the
provider's reasoning trace as user-facing content.

## Recovery

Preserve the existing empty-final truncation recovery. If Gemma consumes the
output budget on thinking and returns no final content, AIDaemon retries and
uses its existing escalation:

1. Reduce reasoning effort on the first retry.
2. Disable reasoning on the second retry.
3. Force a text-only response with reasoning disabled on later retries.

For llama.cpp thinking, a retry-level `"off"` override must suppress the
Gemma-specific thinking fields as well as the generic reasoning field.
Otherwise the existing recovery would claim to disable reasoning while the
chat template continued enabling it.

## Testing

Provider request-body tests verify:

- Default OpenAI-compatible requests omit llama.cpp thinking fields.
- Enabling the setting emits `chat_template_kwargs.enable_thinking = true`
  and `reasoning_format = "deepseek"`.
- A per-call `"off"` reasoning override suppresses those fields.

Provider response tests verify:

- `reasoning_content` is parsed into `ProviderResponse.thinking`.
- Final `content` remains separate and unchanged.

Configuration tests verify the new setting parses and defaults to disabled.
Focused tests run first, followed by formatting, clippy with all features, and
the full test suite according to the repository pre-commit checklist.

## Non-Goals

- Displaying raw model reasoning to users.
- Enabling thinking for fallback providers.
- Inferring provider capabilities from model-name substrings.
- Changing the existing channel status indicators.
