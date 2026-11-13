//! What a user is told when their request waits for the model provider to
//! recover: the turn engine says it, the host keeps the queue.

/// Attempts before the request is abandoned (initial failure counts as 0).
pub const MAX_DEFERRED_ATTEMPTS: u16 = 4;

/// What the user is told when their request was queued (`attempt` 0) or
/// re-queued after another failure.
pub fn deferral_notice(attempt: u16, error: &anyhow::Error) -> String {
    let notice = queued_request_notice(attempt, error);
    match error
        .downcast_ref::<crate::chat_bridge::ProviderError>()
        .and_then(crate::chat_bridge::ProviderError::fallback_failures_note)
    {
        Some(note) => format!("{notice}\n\n{note}"),
        None => notice,
    }
}

fn queued_request_notice(attempt: u16, error: &anyhow::Error) -> String {
    let quota = error
        .downcast_ref::<crate::chat_bridge::ProviderError>()
        .filter(|e| e.kind == crate::chat_bridge::ProviderErrorKind::QuotaExhausted);
    if let Some(quota) = quota {
        return match quota.resets_at {
            Some(at) => format!(
                "The model plan's usage limit has been reached; it resets {}. I saved this request and will run it automatically after the reset, then reply here.",
                crate::chat_bridge::describe_reset(at, std::time::SystemTime::now())
            ),
            None => "The model plan's usage limit has been reached. I saved this request and will retry it automatically, then reply here.".to_string(),
        };
    }
    match attempt {
        0 => "The model provider is unavailable right now. I saved this request and will retry it automatically once the provider recovers, then reply here.".to_string(),
        attempt => format!(
            "The model provider is still unavailable (automatic retry {attempt} of {MAX_DEFERRED_ATTEMPTS}). I will keep retrying with a longer wait and reply here when it succeeds."
        ),
    }
}
