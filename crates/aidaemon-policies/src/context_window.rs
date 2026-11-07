//! View-only context compaction as a `before_model` hook.

use std::collections::HashMap;

use aidaemon_core::{Hook, HookContext, Message, ModelRequest, Role};
use async_trait::async_trait;

/// Rough token estimate: characters / 4, counting content, tool-call names
/// and arguments. Good enough to decide *when* to compact.
pub fn estimate_tokens(messages: &[Message]) -> usize {
    messages
        .iter()
        .map(message_chars)
        .sum::<usize>()
        .div_ceil(4)
}

fn message_chars(m: &Message) -> usize {
    m.content.chars().count()
        + m.tool_calls
            .iter()
            .map(|c| c.name.len() + c.arguments.to_string().len())
            .sum::<usize>()
}

/// Keeps each request under a token budget without touching stored state.
///
/// When the estimate exceeds the budget:
/// 1. System messages, the first user message and the most recent
///    `keep_recent_turns` turns (a turn is an assistant message plus its
///    tool results, and any user message after it) stay intact.
/// 2. Older tool results are replaced, oldest first, with
///    `[tool result for <tool> elided: N chars]` until under budget.
/// 3. If still over, the oldest unprotected units are dropped whole.
///
/// An assistant message with tool calls and its tool results form one unit:
/// they are kept or dropped together, never split (providers reject a tool
/// result without its call and vice versa). If the protected part alone is
/// over budget, the request is sent as is.
#[derive(Debug, Clone)]
pub struct ContextWindow {
    budget_tokens: usize,
    keep_recent_turns: usize,
}

impl ContextWindow {
    pub fn new(budget_tokens: usize) -> Self {
        Self {
            budget_tokens,
            keep_recent_turns: 4,
        }
    }

    pub fn keep_recent_turns(mut self, turns: usize) -> Self {
        self.keep_recent_turns = turns;
        self
    }

    /// Compact `messages` in place. Returns true if anything changed.
    pub fn compact(&self, messages: &mut Vec<Message>) -> bool {
        if estimate_tokens(messages) <= self.budget_tokens {
            return false;
        }
        let units = units(messages);
        let protected = self.protected_units(messages, &units);

        // Tool name per call id, for placeholders.
        let tool_names: HashMap<String, String> = messages
            .iter()
            .flat_map(|m| m.tool_calls.iter())
            .map(|c| (c.id.clone(), c.name.clone()))
            .collect();

        let mut changed = false;
        // Phase 1: elide old tool results, oldest first.
        'elide: for (u, range) in units.iter().enumerate() {
            if protected[u] {
                continue;
            }
            for i in range.clone() {
                if estimate_tokens(messages) <= self.budget_tokens {
                    break 'elide;
                }
                let m = &messages[i];
                if m.role != Role::Tool {
                    continue;
                }
                let tool = m
                    .tool_call_id
                    .as_ref()
                    .and_then(|id| tool_names.get(id))
                    .map_or("unknown", String::as_str);
                let placeholder = format!(
                    "[tool result for {tool} elided: {} chars]",
                    m.content.chars().count()
                );
                if placeholder.len() < m.content.len() {
                    messages[i].content = placeholder;
                    changed = true;
                }
            }
        }

        // Phase 2: drop oldest unprotected units whole.
        if estimate_tokens(messages) > self.budget_tokens {
            let mut keep = vec![true; messages.len()];
            let mut total = estimate_tokens(messages) * 4;
            for (u, range) in units.iter().enumerate() {
                if total.div_ceil(4) <= self.budget_tokens {
                    break;
                }
                if protected[u] {
                    continue;
                }
                for i in range.clone() {
                    keep[i] = false;
                    total = total.saturating_sub(message_chars(&messages[i]));
                }
                changed = true;
            }
            let mut index = 0;
            messages.retain(|_| {
                let k = keep[index];
                index += 1;
                k
            });
        }
        changed
    }

    fn protected_units(&self, messages: &[Message], units: &[std::ops::Range<usize>]) -> Vec<bool> {
        let first_user = messages.iter().position(|m| m.role == Role::User);
        // Start of the Nth most recent assistant unit.
        let recent_start = units
            .iter()
            .filter(|r| messages[r.start].role == Role::Assistant)
            .rev()
            .take(self.keep_recent_turns)
            .last()
            .map_or(messages.len(), |r| r.start);
        let recent_start = if self.keep_recent_turns == 0 {
            messages.len()
        } else {
            recent_start
        };
        units
            .iter()
            .map(|r| {
                r.start >= recent_start
                    || messages[r.start].role == Role::System
                    || first_user.is_some_and(|f| r.contains(&f))
            })
            .collect()
    }
}

/// Partition into atomic units: an assistant message with tool calls plus
/// the tool results that follow it; every other message alone.
fn units(messages: &[Message]) -> Vec<std::ops::Range<usize>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < messages.len() {
        let start = i;
        i += 1;
        if messages[start].role == Role::Assistant && !messages[start].tool_calls.is_empty() {
            while i < messages.len() && messages[i].role == Role::Tool {
                i += 1;
            }
        }
        out.push(start..i);
    }
    out
}

#[async_trait]
impl Hook for ContextWindow {
    async fn before_model(&self, _ctx: &HookContext<'_>, request: &mut ModelRequest) {
        self.compact(&mut request.messages);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_core::ToolCall;
    use serde_json::json;

    fn round(id: &str, result: &str) -> Vec<Message> {
        vec![
            Message::assistant(
                "",
                vec![ToolCall::new(id, "read_file", json!({"path": id}))],
            ),
            Message::tool_result(id, result),
        ]
    }

    fn transcript(rounds: usize, result_len: usize) -> Vec<Message> {
        let mut m = vec![
            Message::system("sys"),
            Message::user("Summarize the files for Alice Rivera"),
        ];
        for r in 0..rounds {
            m.extend(round(&format!("c{r}"), &"x".repeat(result_len)));
        }
        m.push(Message::assistant("almost done", vec![]));
        m
    }

    /// Every tool result is preceded (within its unit) by the call that
    /// produced it, and every call has its result.
    fn well_formed(messages: &[Message]) -> bool {
        let mut open: Vec<String> = Vec::new();
        for m in messages {
            match m.role {
                Role::Tool => {
                    let id = m.tool_call_id.clone().unwrap();
                    match open.iter().position(|o| *o == id) {
                        Some(p) => {
                            open.remove(p);
                        }
                        None => return false,
                    }
                }
                _ => {
                    if !open.is_empty() {
                        return false;
                    }
                    open = m.tool_calls.iter().map(|c| c.id.clone()).collect();
                }
            }
        }
        open.is_empty()
    }

    #[test]
    fn under_budget_is_untouched() {
        let mut m = transcript(3, 100);
        let before = m.clone();
        assert!(!ContextWindow::new(10_000).compact(&mut m));
        assert_eq!(m, before);
    }

    #[test]
    fn old_tool_results_are_elided_first() {
        let mut m = transcript(10, 4_000);
        let window = ContextWindow::new(6_000).keep_recent_turns(2);
        assert!(window.compact(&mut m));
        assert!(estimate_tokens(&m) <= 6_000);
        assert_eq!(m.len(), transcript(10, 4_000).len(), "nothing dropped");
        assert_eq!(
            m[3].content,
            "[tool result for read_file elided: 4000 chars]"
        );
        // The last two turns (round c9 and the final answer) are intact.
        let n = m.len();
        assert_eq!(m[n - 2].content.len(), 4_000);
        assert!(well_formed(&m));
    }

    #[test]
    fn whole_units_are_dropped_when_eliding_is_not_enough() {
        let mut m = Vec::new();
        m.push(Message::system("sys"));
        m.push(Message::user("first request"));
        for r in 0..20 {
            m.push(Message::user(format!("follow-up {r} {}", "y".repeat(400))));
            m.extend(round(&format!("c{r}"), "short"));
        }
        let window = ContextWindow::new(400).keep_recent_turns(2);
        assert!(window.compact(&mut m));
        assert!(estimate_tokens(&m) <= 400, "{}", estimate_tokens(&m));
        assert_eq!(m[0].content, "sys");
        assert_eq!(m[1].content, "first request");
        assert!(well_formed(&m));
        assert_eq!(m.last().unwrap().tool_call_id.as_deref(), Some("c19"));
    }

    #[test]
    fn protected_part_over_budget_is_sent_as_is() {
        let mut m = vec![Message::system("s".repeat(10_000)), Message::user("hi")];
        let before = m.clone();
        ContextWindow::new(10).compact(&mut m);
        assert_eq!(m, before);
    }
}
