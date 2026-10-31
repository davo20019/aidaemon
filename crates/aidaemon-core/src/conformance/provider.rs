use std::time::Duration;

use futures::StreamExt;
use serde_json::json;

use super::ConformanceReport;
use crate::message::Message;
use crate::provider::{
    collect_stream, ModelRequest, ModelResponse, Provider, StreamEvent, ToolChoice, ToolSpec,
};

/// Stable names of the checks run by [`check_provider`]. Per-request checks
/// are suffixed with the probe label: `/plain` or `/with_tool`.
pub mod provider_checks {
    /// `name()` is not empty (it appears in events and traces).
    pub const NAME_NON_EMPTY: &str = "provider.name_non_empty";
    /// `stream` yields zero or more `TextDelta`s, then exactly one `Done`,
    /// then ends: no error, no second `Done`, nothing after `Done`.
    pub const STREAM_WELL_FORMED: &str = "provider.stream_well_formed";
    /// When the stream yields deltas, their concatenation equals
    /// `Done.text`, so streamed and final text never disagree.
    pub const DELTAS_MATCH_DONE_TEXT: &str = "provider.deltas_match_done_text";
    /// `complete` and `collect_stream(stream)` succeed for the same request
    /// and agree on text and tool calls.
    pub const COMPLETE_MATCHES_STREAM: &str = "provider.complete_matches_stream";
    /// Tool call ids are unique within a response when present. Empty ids
    /// are allowed: the loop fills them in as `call_{turn}_{index}` (a note
    /// is recorded).
    pub const TOOL_CALL_IDS: &str = "provider.tool_call_ids";
    /// Every tool call has a non-empty name and arguments that are parsed
    /// JSON, not a JSON string holding JSON (providers parse wire strings;
    /// unparseable input is kept as an object such as `{"_raw": ...}`).
    pub const TOOL_CALL_SHAPE: &str = "provider.tool_call_shape";
    /// Usage is consistent: `cached_input_tokens <= input_tokens` and
    /// `billable_tokens() <= input_tokens + output_tokens`. (Fields are
    /// unsigned, so non-negativity holds by construction.)
    pub const USAGE_CONSISTENT: &str = "provider.usage_consistent";
}

/// Calls [`check_provider`] makes per probe request, in this order:
/// `stream` (inspected event by event), `complete`, `stream` again (folded
/// with [`collect_stream`]). Script mock servers accordingly.
pub const PROVIDER_CALLS_PER_REQUEST: usize = 3;

/// How long one provider call may take before the kit gives up on it.
pub const PROVIDER_CALL_TIMEOUT: Duration = Duration::from_secs(30);

/// The probe requests [`check_provider`] sends, in order, with their labels:
/// `plain` (no tools) then `with_tool` (one `get_time` function). Each is
/// sent [`PROVIDER_CALLS_PER_REQUEST`] times, so the kit makes six calls.
pub fn probe_requests() -> Vec<(&'static str, ModelRequest)> {
    let base = |tools: Vec<ToolSpec>, prompt: &str| ModelRequest {
        model: "conformance-probe".into(),
        messages: vec![
            Message::system("You are a conformance probe. Keep answers short."),
            Message::user(prompt),
        ],
        tools,
        tool_choice: ToolChoice::Auto,
        max_output_tokens: Some(256),
    };
    vec![
        ("plain", base(Vec::new(), "Reply with a short greeting.")),
        (
            "with_tool",
            base(
                vec![ToolSpec {
                    name: "get_time".into(),
                    description: "Current local time in a city".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": false
                    }),
                }],
                "What time is it in Springfield? Use the get_time tool.",
            ),
        ),
    ]
}

/// Run every provider check (see [`provider_checks`]) against `provider`.
///
/// The provider must answer identical requests identically, because the kit
/// compares `complete` with a separately collected stream: script a
/// deterministic backend, for example
/// `ScriptedProvider::new([]).then_repeat(reply)`, or a mock server queued
/// with [`PROVIDER_CALLS_PER_REQUEST`] copies of each reply (see
/// [`probe_requests`] for the order). Needs a Tokio runtime with timers.
pub async fn check_provider(provider: &dyn Provider) -> ConformanceReport {
    use provider_checks::*;

    let name = provider.name().to_string();
    let mut report = ConformanceReport::new(if name.is_empty() {
        "<unnamed provider>".to_string()
    } else {
        name.clone()
    });
    report.record(
        NAME_NON_EMPTY,
        if name.trim().is_empty() {
            Err("name() is empty".into())
        } else {
            Ok(())
        },
    );

    for (label, request) in probe_requests() {
        let check = |base: &str| format!("{base}/{label}");
        let mut responses: Vec<(&str, ModelResponse)> = Vec::new();

        // 1. Inspect the stream event by event.
        match drain(provider, &request).await {
            Err(reason) => {
                report.fail(check(STREAM_WELL_FORMED), reason);
                report.fail(
                    check(DELTAS_MATCH_DONE_TEXT),
                    "skipped: the stream was not well formed",
                );
            }
            Ok(events) => match well_formed(&events) {
                Err(reason) => {
                    report.fail(check(STREAM_WELL_FORMED), reason);
                    report.fail(
                        check(DELTAS_MATCH_DONE_TEXT),
                        "skipped: the stream was not well formed",
                    );
                }
                Ok((deltas, done)) => {
                    report.pass(check(STREAM_WELL_FORMED));
                    if deltas.is_empty() {
                        report.pass(check(DELTAS_MATCH_DONE_TEXT));
                        report.note(format!(
                            "{}: no text deltas were streamed",
                            check(DELTAS_MATCH_DONE_TEXT)
                        ));
                    } else {
                        let joined: String = deltas.concat();
                        report.record(
                            check(DELTAS_MATCH_DONE_TEXT),
                            if joined == done.text {
                                Ok(())
                            } else {
                                Err(format!(
                                    "deltas joined to {joined:?} but Done.text is {:?}",
                                    done.text
                                ))
                            },
                        );
                    }
                    responses.push(("stream", done));
                }
            },
        }

        // 2. complete vs collect_stream(stream).
        let completed = timed(provider.complete(&request)).await;
        let collected = timed(collect_stream(provider.stream(&request))).await;
        match (completed, collected) {
            (Ok(a), Ok(b)) => {
                let mut diffs = Vec::new();
                if a.text != b.text {
                    diffs.push(format!(
                        "text: complete {:?} vs stream {:?}",
                        a.text, b.text
                    ));
                }
                if a.tool_calls != b.tool_calls {
                    diffs.push(format!(
                        "tool calls: complete {:?} vs stream {:?}",
                        a.tool_calls, b.tool_calls
                    ));
                }
                report.record(
                    check(COMPLETE_MATCHES_STREAM),
                    if diffs.is_empty() {
                        Ok(())
                    } else {
                        Err(diffs.join("; "))
                    },
                );
                responses.push(("complete", a));
                responses.push(("collect_stream", b));
            }
            (a, b) => {
                let mut errors = Vec::new();
                if let Err(e) = a {
                    errors.push(format!("complete failed: {e}"));
                }
                if let Err(e) = b {
                    errors.push(format!("collect_stream failed: {e}"));
                }
                report.fail(check(COMPLETE_MATCHES_STREAM), errors.join("; "));
            }
        }

        // 3. Response-level checks over every response obtained.
        if responses.is_empty() {
            for base in [TOOL_CALL_IDS, TOOL_CALL_SHAPE, USAGE_CONSISTENT] {
                report.fail(check(base), "skipped: no response was obtained");
            }
            continue;
        }
        let mut id_problems = Vec::new();
        let mut shape_problems = Vec::new();
        let mut usage_problems = Vec::new();
        let mut empty_ids = false;
        let mut saw_calls = false;
        for (source, response) in &responses {
            let mut seen = std::collections::HashSet::new();
            for call in &response.tool_calls {
                saw_calls = true;
                if call.id.is_empty() {
                    empty_ids = true;
                } else if !seen.insert(call.id.as_str()) {
                    id_problems.push(format!("{source}: duplicate tool call id {:?}", call.id));
                }
                if call.name.trim().is_empty() {
                    shape_problems.push(format!("{source}: tool call {:?} has no name", call.id));
                }
                if call.arguments.is_string() {
                    shape_problems.push(format!(
                        "{source}: arguments of {:?} are a JSON string; parse them first",
                        call.name
                    ));
                }
            }
            let u = response.usage;
            if u.cached_input_tokens > u.input_tokens {
                usage_problems.push(format!(
                    "{source}: cached_input_tokens {} > input_tokens {}",
                    u.cached_input_tokens, u.input_tokens
                ));
            }
            if u.billable_tokens() > u.input_tokens.saturating_add(u.output_tokens) {
                usage_problems.push(format!(
                    "{source}: billable_tokens {} > input + output",
                    u.billable_tokens()
                ));
            }
        }
        if empty_ids {
            report.note(format!(
                "{}: provider left tool call ids empty; the loop fills them as call_{{turn}}_{{index}}",
                check(TOOL_CALL_IDS)
            ));
        }
        if !saw_calls {
            report.note(format!(
                "{label}: the backend returned no tool calls, so tool call checks were vacuous"
            ));
        }
        for (base, problems) in [
            (TOOL_CALL_IDS, id_problems),
            (TOOL_CALL_SHAPE, shape_problems),
            (USAGE_CONSISTENT, usage_problems),
        ] {
            report.record(
                check(base),
                if problems.is_empty() {
                    Ok(())
                } else {
                    Err(problems.join("; "))
                },
            );
        }
    }
    report
}

async fn timed<T>(
    fut: impl std::future::Future<Output = Result<T, crate::ProviderError>>,
) -> Result<T, String> {
    match tokio::time::timeout(PROVIDER_CALL_TIMEOUT, fut).await {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err(format!("timed out after {PROVIDER_CALL_TIMEOUT:?}")),
    }
}

/// Drain a whole stream (past `Done`, to catch trailing events).
async fn drain(
    provider: &dyn Provider,
    request: &ModelRequest,
) -> Result<Vec<StreamEvent>, String> {
    let collect = async {
        let mut stream = provider.stream(request);
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok(event) => events.push(event),
                Err(e) => {
                    return Err(format!(
                        "stream yielded an error after {} event(s): {e}",
                        events.len()
                    ))
                }
            }
        }
        Ok(events)
    };
    tokio::time::timeout(PROVIDER_CALL_TIMEOUT, collect)
        .await
        .unwrap_or_else(|_| {
            Err(format!(
                "stream did not end within {PROVIDER_CALL_TIMEOUT:?}"
            ))
        })
}

fn well_formed(events: &[StreamEvent]) -> Result<(Vec<String>, ModelResponse), String> {
    let mut deltas = Vec::new();
    let mut done: Option<ModelResponse> = None;
    for (i, event) in events.iter().enumerate() {
        match (event, &done) {
            (StreamEvent::TextDelta(d), None) => deltas.push(d.clone()),
            (StreamEvent::Done(r), None) => done = Some(r.clone()),
            (StreamEvent::TextDelta(_), Some(_)) => {
                return Err(format!("event {i} is a TextDelta after Done"))
            }
            (StreamEvent::Done(_), Some(_)) => return Err(format!("event {i} is a second Done")),
        }
    }
    match done {
        Some(done) => Ok((deltas, done)),
        None => Err(format!(
            "stream ended after {} event(s) without a Done",
            events.len()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{ProviderError, Usage};
    use crate::testing::{call, text, tool_calls, ScriptedProvider, StreamingProvider};
    use futures::stream::{self, BoxStream};

    #[tokio::test]
    async fn scripted_providers_conform() {
        let reply = text("Hello there, friend.");
        check_provider(&ScriptedProvider::new([]).then_repeat(reply.clone()))
            .await
            .assert_all_passed();
        let report = check_provider(&StreamingProvider(
            ScriptedProvider::new([]).then_repeat(reply),
        ))
        .await;
        report.assert_all_passed();
        assert_eq!(report.passed.len(), 1 + 2 * 6);
    }

    /// Streams deltas that disagree with Done, then a trailing delta.
    struct Broken;

    #[async_trait::async_trait]
    impl Provider for Broken {
        fn name(&self) -> &str {
            "broken"
        }
        async fn complete(&self, _: &ModelRequest) -> Result<ModelResponse, ProviderError> {
            Ok(ModelResponse {
                text: "different".into(),
                tool_calls: vec![call("", "get_time", json!("{\"city\":\"x\"}"))],
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cached_input_tokens: 5,
                    cache_creation_tokens: 0,
                },
            })
        }
        fn stream<'a>(
            &'a self,
            _: &'a ModelRequest,
        ) -> BoxStream<'a, Result<StreamEvent, ProviderError>> {
            Box::pin(stream::iter([
                Ok(StreamEvent::TextDelta("hel".into())),
                Ok(StreamEvent::Done(tool_calls([]))),
                Ok(StreamEvent::TextDelta("late".into())),
            ]))
        }
    }

    #[tokio::test]
    async fn broken_provider_reports_every_problem() {
        let report = check_provider(&Broken).await;
        for check in [
            "provider.stream_well_formed/plain",
            "provider.complete_matches_stream/plain",
            "provider.tool_call_shape/with_tool",
            "provider.usage_consistent/plain",
        ] {
            assert!(report.failure(check).is_some(), "{check}: {report}");
        }
        assert!(report.passed_check("provider.tool_call_ids/plain"));
        assert!(report.notes.iter().any(|n| n.contains("call_{turn}")));
    }
}
