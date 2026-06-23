//! Concurrent prefetch for read-only tool-call batches.
//!
//! When the model emits several tool calls in one assistant message, the
//! execution loop processes them strictly sequentially — so three slow
//! network reads cost the sum of their latencies. This module pre-executes
//! the I/O for batches that are provably safe to overlap (every call
//! read-only, no approvals, distinct arguments) while the sequential loop
//! keeps full ownership of guards, budgets, and result processing.
//!
//! Correctness is self-verifying: results are keyed by call id together
//! with the exact arguments the prefetch executed. The sequential loop
//! consumes a prefetched result only when its own computed effective
//! arguments match byte-for-byte; any divergence (e.g. project-dir
//! injection rewrote a path) falls back to live execution and the spare
//! read-only result is discarded.

use super::execution_io::{execute_tool_call_io, ToolExecutionIoCtx, ToolExecutionIoResult};
use crate::agent::*;
use crate::execution_policy::PolicyBundle;

pub(super) struct PrefetchedIo {
    /// Arguments the prefetch executed with — must match the loop's
    /// computed effective arguments for the result to be consumed.
    pub arguments: String,
    pub io: ToolExecutionIoResult,
}

/// Shared per-iteration context for building each call's I/O context.
pub(super) struct PrefetchCtx<'a> {
    pub model: &'a str,
    pub idempotency_key: Option<&'a str>,
    pub project_scope: Option<&'a str>,
    pub session_id: &'a str,
    pub task_id: &'a str,
    pub status_tx: &'a Option<mpsc::Sender<StatusUpdate>>,
    pub channel_ctx: &'a ChannelContext,
    pub user_role: UserRole,
    pub heartbeat: &'a Option<Arc<AtomicU64>>,
    pub emitter: &'a crate::events::EventEmitter,
    pub policy_bundle: &'a PolicyBundle,
}

/// A batch qualifies for concurrent prefetch only when overlap is provably
/// safe: 2+ calls, every tool known and read-only without approval, no
/// MCP tools (unknown semantics), nothing in cooldown, and all
/// (name, arguments) pairs distinct (identical calls must stay sequential
/// so the repetition guards see them).
pub(super) fn batch_is_prefetch_eligible(
    tool_calls: &[ToolCall],
    capabilities: &HashMap<String, ToolCapabilities>,
    unknown_tools: &HashSet<String>,
    cooldowns: &HashMap<String, usize>,
    iteration: usize,
) -> bool {
    if tool_calls.len() < 2 {
        return false;
    }
    let mut seen = HashSet::new();
    tool_calls.iter().all(|tc| {
        let Some(caps) = capabilities.get(&tc.name) else {
            return false;
        };
        caps.read_only
            && !caps.needs_approval
            && !tc.name.contains("__")
            && !unknown_tools.contains(&tc.name)
            && cooldowns
                .get(&tc.name)
                .is_none_or(|until| iteration > *until)
            && seen.insert((tc.name.clone(), tc.arguments.clone()))
    })
}

/// Execute the batch's I/O concurrently. Guards and result processing stay
/// in the sequential loop; this only overlaps the tool-call latency.
pub(super) async fn prefetch_read_only_batch(
    agent: &Agent,
    tool_calls: &[ToolCall],
    ctx: &PrefetchCtx<'_>,
) -> HashMap<String, PrefetchedIo> {
    let futures = tool_calls.iter().map(|tc| async move {
        let io_ctx = ToolExecutionIoCtx {
            effective_arguments: &tc.arguments,
            model: ctx.model,
            idempotency_key: ctx.idempotency_key,
            injected_project_dir: None,
            project_scope: ctx.project_scope,
            session_id: ctx.session_id,
            task_id: ctx.task_id,
            status_tx: ctx.status_tx,
            channel_ctx: ctx.channel_ctx,
            user_role: ctx.user_role,
            heartbeat: ctx.heartbeat,
            emitter: ctx.emitter,
            policy_bundle: ctx.policy_bundle,
            correction_preapproved: false,
            suppress_trusted_session: false,
        };
        let io = execute_tool_call_io(agent, tc, &io_ctx).await;
        (
            tc.id.clone(),
            PrefetchedIo {
                arguments: tc.arguments.clone(),
                io,
            },
        )
    });
    futures::future::join_all(futures)
        .await
        .into_iter()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(id: &str, name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments: args.to_string(),
            extra_content: None,
        }
    }

    fn read_only_caps() -> ToolCapabilities {
        ToolCapabilities {
            read_only: true,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }

    fn caps_for(names: &[&str]) -> HashMap<String, ToolCapabilities> {
        names
            .iter()
            .map(|n| (n.to_string(), read_only_caps()))
            .collect()
    }

    #[test]
    fn eligible_batch_of_distinct_read_only_calls() {
        let calls = vec![
            call("1", "web_fetch", r#"{"url":"https://a.example"}"#),
            call("2", "web_fetch", r#"{"url":"https://b.example"}"#),
        ];
        assert!(batch_is_prefetch_eligible(
            &calls,
            &caps_for(&["web_fetch"]),
            &HashSet::new(),
            &HashMap::new(),
            3,
        ));
    }

    #[test]
    fn single_call_is_not_eligible() {
        let calls = vec![call("1", "web_fetch", "{}")];
        assert!(!batch_is_prefetch_eligible(
            &calls,
            &caps_for(&["web_fetch"]),
            &HashSet::new(),
            &HashMap::new(),
            1,
        ));
    }

    #[test]
    fn mutating_or_approval_tools_disqualify_the_whole_batch() {
        let mut caps = caps_for(&["read_file"]);
        caps.insert(
            "write_file".to_string(),
            ToolCapabilities {
                read_only: false,
                external_side_effect: false,
                needs_approval: false,
                idempotent: false,
                high_impact_write: true,
            },
        );
        let calls = vec![
            call("1", "read_file", r#"{"path":"a"}"#),
            call("2", "write_file", r#"{"path":"b"}"#),
        ];
        assert!(!batch_is_prefetch_eligible(
            &calls,
            &caps,
            &HashSet::new(),
            &HashMap::new(),
            1,
        ));

        let mut approval_caps = caps_for(&["read_file"]);
        approval_caps.insert(
            "guarded_read".to_string(),
            ToolCapabilities {
                needs_approval: true,
                ..read_only_caps()
            },
        );
        let calls = vec![
            call("1", "read_file", r#"{"path":"a"}"#),
            call("2", "guarded_read", r#"{"path":"b"}"#),
        ];
        assert!(!batch_is_prefetch_eligible(
            &calls,
            &approval_caps,
            &HashSet::new(),
            &HashMap::new(),
            1,
        ));
    }

    #[test]
    fn unknown_tool_or_missing_capabilities_disqualify() {
        let calls = vec![
            call("1", "web_fetch", r#"{"url":"a"}"#),
            call("2", "made_up_tool", "{}"),
        ];
        assert!(!batch_is_prefetch_eligible(
            &calls,
            &caps_for(&["web_fetch"]),
            &HashSet::new(),
            &HashMap::new(),
            1,
        ));

        let mut unknown = HashSet::new();
        unknown.insert("web_fetch".to_string());
        let calls = vec![
            call("1", "web_fetch", r#"{"url":"a"}"#),
            call("2", "web_fetch", r#"{"url":"b"}"#),
        ];
        assert!(!batch_is_prefetch_eligible(
            &calls,
            &caps_for(&["web_fetch"]),
            &unknown,
            &HashMap::new(),
            1,
        ));
    }

    #[test]
    fn duplicate_calls_and_cooldowns_disqualify() {
        // Identical (name, args) must stay sequential for repetition guards.
        let calls = vec![
            call("1", "web_fetch", r#"{"url":"same"}"#),
            call("2", "web_fetch", r#"{"url":"same"}"#),
        ];
        assert!(!batch_is_prefetch_eligible(
            &calls,
            &caps_for(&["web_fetch"]),
            &HashSet::new(),
            &HashMap::new(),
            1,
        ));

        let mut cooldowns = HashMap::new();
        cooldowns.insert("web_fetch".to_string(), 5usize);
        let calls = vec![
            call("1", "web_fetch", r#"{"url":"a"}"#),
            call("2", "web_fetch", r#"{"url":"b"}"#),
        ];
        // iteration 4 <= cooldown-until 5 → blocked
        assert!(!batch_is_prefetch_eligible(
            &calls,
            &caps_for(&["web_fetch"]),
            &HashSet::new(),
            &cooldowns,
            4,
        ));
        // cooldown expired → eligible again
        assert!(batch_is_prefetch_eligible(
            &calls,
            &caps_for(&["web_fetch"]),
            &HashSet::new(),
            &cooldowns,
            6,
        ));
    }

    #[test]
    fn mcp_tools_disqualify() {
        let calls = vec![
            call("1", "server__lookup", "{}"),
            call("2", "server__search", "{}"),
        ];
        assert!(!batch_is_prefetch_eligible(
            &calls,
            &caps_for(&["server__lookup", "server__search"]),
            &HashSet::new(),
            &HashMap::new(),
            1,
        ));
    }
}
