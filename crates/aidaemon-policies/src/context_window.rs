//! View-only context compaction as a `before_model` hook.

use std::collections::HashMap;
use std::ops::Range;

use aidaemon_core::{Hook, HookContext, Message, ModelRequest, Role, ToolCall, ToolSpec};
use async_trait::async_trait;

/// Rough token estimate: characters / 4, counting content, tool-call names
/// and arguments, and a fixed allowance per image and audio clip. Good
/// enough to decide *when* to compact.
pub fn estimate_tokens(messages: &[Message]) -> usize {
    messages
        .iter()
        .map(message_chars)
        .sum::<usize>()
        .div_ceil(4)
}

/// What one image costs a request, in estimate characters (about 1.2k
/// tokens: a screenshot, not its base64 bytes).
const IMAGE_CHARS: usize = 4_800;

/// What an audio clip costs: about a token per 100 bytes of audio (the
/// data is base64, 4 characters per 3 bytes), at least 64 tokens.
fn audio_chars(data: &str) -> usize {
    ((data.len() * 3 / 4) / 100).max(64) * 4
}

fn message_chars(m: &Message) -> usize {
    m.content.chars().count()
        + m.tool_calls
            .iter()
            .map(|c| c.name.len() + c.arguments.to_string().len())
            .sum::<usize>()
        + m.images.len() * IMAGE_CHARS
        + m.audio.iter().map(|a| audio_chars(&a.data)).sum::<usize>()
}

/// Characters a request spends on its tool definitions, which the
/// provider sends with every call.
pub fn tool_spec_chars(tools: &[ToolSpec]) -> usize {
    tools
        .iter()
        .map(|t| t.name.len() + t.description.chars().count() + t.parameters.to_string().len())
        .sum()
}

/// Longest arguments and quoted output line a stub carries.
const STUB_ARGS_CHARS: usize = 200;
const STUB_LINE_CHARS: usize = 120;

/// How each message of the transcript goes into one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shown {
    Full,
    Stub,
    Dropped,
}

/// Keeps each request under a token budget without touching stored state,
/// and keeps the request's prefix stable so the provider's prompt cache
/// keeps hitting.
///
/// The budget covers what the provider is sent: every message plus the
/// tool definitions ([`ContextWindow::compact_request`]), and anything else
/// the caller sends around the messages
/// ([`ContextWindow::compact_with_overhead`]). Output tokens are not
/// included; set the budget below the model's window by at least the
/// output limit.
///
/// **Coarse, stable steps.** Old tool results are replaced by a short stub
/// up to a *cut*, and units before a *drop* point are left out. Both move
/// only when a request would pass the budget. The cut then moves far
/// enough to bring the request down to half the budget, so the following
/// requests grow back on the same prefix until the budget is reached again;
/// stubbed units are dropped (oldest first) only once the stubs themselves
/// fill a quarter of the budget, since they keep what the model already
/// learned (its own notes, which calls it made) at little cost. The two
/// points are replayed from the transcript itself (every earlier model
/// call saw a prefix of it), so the hook keeps no state and a resumed run
/// computes the same view.
///
/// **Stubs say what was there.** An elided result becomes
/// `[earlier tool result elided ...: <tool> <arguments> returned N chars
/// (first line: "..."; last line: "...")...]`. A result an earlier request
/// showed in full says so, and that the call needs repeating only for its
/// exact content. A result no request has shown yet (it came in a turn
/// whose results did not all fit) says the model has not seen it and must
/// call the tool again, with fewer calls at once: a repeated call is a
/// new, recent result, which the next request shows in full.
///
/// **Never over budget.** System messages, injected harness messages and
/// the user's first and latest messages are never elided or dropped, and the most
/// recent `keep_recent_turns` turns (default 1: the results the model has
/// not seen yet) are kept whole by the
/// stable steps. If the request still passes the budget (the recent turns
/// alone are too big), this request alone also elides recent results,
/// oldest first, and then drops the oldest units until it fits. It is sent
/// over budget only if the never-touched part alone is.
///
/// An assistant message with tool calls and its tool results form one unit:
/// they are kept or dropped together, never split (providers reject a tool
/// result without its call and vice versa).
#[derive(Debug, Clone)]
pub struct ContextWindow {
    budget_tokens: usize,
    keep_recent_turns: usize,
}

impl ContextWindow {
    pub fn new(budget_tokens: usize) -> Self {
        Self {
            budget_tokens,
            keep_recent_turns: 1,
        }
    }

    /// The token budget each request is kept under.
    pub fn budget_tokens(&self) -> usize {
        self.budget_tokens
    }

    /// Turns the stable steps keep whole (a turn is an assistant message
    /// plus its tool results, and any user message after it). More turns
    /// mean smaller steps, so the prefix changes more often.
    pub fn keep_recent_turns(mut self, turns: usize) -> Self {
        self.keep_recent_turns = turns;
        self
    }

    /// Compact `messages` in place, counting messages only. Returns true if
    /// anything changed.
    pub fn compact(&self, messages: &mut Vec<Message>) -> bool {
        let before = messages.clone();
        self.compact_with_overhead(messages, 0);
        *messages != before
    }

    /// Compact a whole request: its messages, with its tool definitions
    /// counted against the budget too. Returns true if anything changed.
    pub fn compact_request(&self, request: &mut ModelRequest) -> bool {
        let overhead = tool_spec_chars(&request.tools);
        let before = request.messages.clone();
        self.compact_with_overhead(&mut request.messages, overhead);
        request.messages != before
    }

    /// Compact `messages` in place, with `overhead_chars` (estimate
    /// characters, 4 to a token) counted for whatever else the request
    /// sends: its tool definitions, a fixed prefix sent ahead of these
    /// messages. Returns, for each message as passed, whether it is still
    /// there (stubbed or whole); the survivors keep their order.
    pub fn compact_with_overhead(
        &self,
        messages: &mut Vec<Message>,
        overhead_chars: usize,
    ) -> Vec<bool> {
        let view = self.view(messages, overhead_chars);
        let calls = calls_by_id(messages);
        let kept: Vec<bool> = view.iter().map(|(s, _)| *s != Shown::Dropped).collect();
        for (m, (shown, seen)) in messages.iter_mut().zip(&view) {
            if *shown == Shown::Stub {
                apply_stub(m, &calls, *seen);
            }
        }
        let mut index = 0;
        messages.retain(|_| {
            let k = kept[index];
            index += 1;
            k
        });
        kept
    }

    /// How each message goes into this request, and whether an earlier
    /// request showed it in full.
    fn view(&self, messages: &[Message], overhead_chars: usize) -> Vec<(Shown, bool)> {
        let budget = self.budget_tokens.saturating_mul(4);
        let low_water = budget / 2;
        let current = overhead_chars + messages.iter().map(message_chars).sum::<usize>();
        // Every earlier request was a prefix of this one, so none of them
        // passed the budget either: nothing was ever cut.
        if current <= budget {
            return vec![(Shown::Full, true); messages.len()];
        }

        let units = units(messages);
        let fixed = fixed_units(messages, &units);
        let calls = calls_by_id(messages);
        let full: Vec<usize> = units
            .iter()
            .map(|r| messages[r.clone()].iter().map(message_chars).sum())
            .collect();
        // The stable steps size stubs as shown ones; the exact view below
        // counts each stub as it is sent.
        let stubbed: Vec<usize> = units
            .iter()
            .enumerate()
            .map(|(u, r)| {
                if fixed[u] {
                    return full[u];
                }
                messages[r.clone()]
                    .iter()
                    .map(|m| match stub_for(m, &calls, true) {
                        Some(stub) => stub.chars().count(),
                        None => message_chars(m),
                    })
                    .sum()
            })
            .collect();
        let dropped: Vec<usize> = (0..units.len())
            .map(|u| if fixed[u] { full[u] } else { 0 })
            .collect();

        // Replay the stable steps at every point a model call was made: the
        // end of each unit an assistant message follows, and now. Each
        // earlier call's exact view says which results it showed in full.
        let assistant_units: Vec<usize> = (0..units.len())
            .filter(|&u| messages[units[u].start].role == Role::Assistant)
            .collect();
        let (mut cut, mut drop) = (0usize, 0usize);
        let mut seen = vec![false; messages.len()];
        let mut view = Vec::new();
        for p in 1..=units.len() {
            if p < units.len() && messages[units[p].start].role != Role::Assistant {
                continue;
            }
            let mut size = overhead_chars
                + dropped[..drop].iter().sum::<usize>()
                + stubbed[drop..cut].iter().sum::<usize>()
                + full[cut..p].iter().sum::<usize>();
            if size > budget {
                let boundary = self.protected_start(&assistant_units, p).max(cut);
                while cut < boundary && size > low_water {
                    size -= full[cut] - stubbed[cut];
                    cut += 1;
                }
                let stub_floor = budget / 4;
                let mut stubs: usize = stubbed[drop..cut].iter().sum::<usize>()
                    - dropped[drop..cut].iter().sum::<usize>();
                while drop < cut && size > low_water && stubs > stub_floor {
                    let freed = stubbed[drop] - dropped[drop];
                    size -= freed;
                    stubs -= freed;
                    drop += 1;
                }
            }
            view = self.view_at(
                messages,
                &units,
                &fixed,
                &calls,
                &seen,
                (p, cut, drop),
                overhead_chars,
            );
            if p < units.len() {
                for (i, shown) in view.iter().enumerate() {
                    if *shown == Shown::Full {
                        seen[i] = true;
                    }
                }
            }
        }
        view.into_iter().zip(seen).collect()
    }

    /// The exact view sent at point `p` (the first `p` units) with the
    /// stable steps at `cut` and `drop`: over the budget, the recent
    /// results are elided oldest first, then the oldest units dropped.
    #[allow(clippy::too_many_arguments)]
    fn view_at(
        &self,
        messages: &[Message],
        units: &[Range<usize>],
        fixed: &[bool],
        calls: &HashMap<String, ToolCall>,
        seen: &[bool],
        (p, cut, drop): (usize, usize, usize),
        overhead_chars: usize,
    ) -> Vec<Shown> {
        let budget = self.budget_tokens.saturating_mul(4);
        let end = units[p - 1].end;
        let mut shown = vec![Shown::Full; end];
        let sent = |i: usize, how: Shown| match how {
            Shown::Full => message_chars(&messages[i]),
            Shown::Stub => stub_for(&messages[i], calls, seen[i])
                .map_or_else(|| message_chars(&messages[i]), |s| s.chars().count()),
            Shown::Dropped => 0,
        };
        let mut size = overhead_chars;
        for (u, range) in units[..p].iter().enumerate() {
            for i in range.clone() {
                if !fixed[u] && u < drop {
                    shown[i] = Shown::Dropped;
                } else if !fixed[u] && u < cut && stub_for(&messages[i], calls, seen[i]).is_some() {
                    shown[i] = Shown::Stub;
                }
                size += sent(i, shown[i]);
            }
        }
        // Still over: the recent turns alone pass the budget. For this
        // request only, elide their results oldest first...
        if size > budget {
            'elide: for range in &units[cut..p] {
                for i in range.clone() {
                    if size <= budget {
                        break 'elide;
                    }
                    if shown[i] == Shown::Full && stub_for(&messages[i], calls, seen[i]).is_some() {
                        size -= sent(i, Shown::Full) - sent(i, Shown::Stub);
                        shown[i] = Shown::Stub;
                    }
                }
            }
        }
        // ...then drop the oldest units, never the newest one.
        if size > budget {
            for (u, range) in units[..p].iter().enumerate().take(p - 1) {
                if size <= budget {
                    break;
                }
                if fixed[u] || shown[range.start] == Shown::Dropped {
                    continue;
                }
                for i in range.clone() {
                    size -= sent(i, shown[i]);
                    shown[i] = Shown::Dropped;
                }
            }
        }
        shown
    }

    /// First unit of the `keep_recent_turns` most recent turns before `p`.
    fn protected_start(&self, assistant_units: &[usize], p: usize) -> usize {
        if self.keep_recent_turns == 0 {
            return p;
        }
        let before = assistant_units.partition_point(|&u| u < p);
        before
            .checked_sub(self.keep_recent_turns)
            .map_or(0, |i| assistant_units[i])
    }
}

/// Partition into atomic units: an assistant message with tool calls plus
/// the tool results that follow it; every other message alone.
fn units(messages: &[Message]) -> Vec<Range<usize>> {
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

/// Units never elided or dropped: system messages, messages the harness or
/// a hook injected, the user's first message (the task) and the user's
/// latest message (the question being answered now). The latest message
/// stays the same through a run's tool rounds, so pinning it keeps the
/// prefix stable; when the user speaks again, the one before it becomes
/// droppable like any older message.
fn fixed_units(messages: &[Message], units: &[Range<usize>]) -> Vec<bool> {
    let is_user = |m: &Message| m.role == Role::User && !m.is_injected();
    let first_user = messages.iter().position(is_user);
    let latest_user = messages.iter().rposition(is_user);
    units
        .iter()
        .map(|r| {
            let m = &messages[r.start];
            m.role == Role::System
                || (m.role == Role::User && m.is_injected())
                || first_user.is_some_and(|f| r.contains(&f))
                || latest_user.is_some_and(|l| r.contains(&l))
        })
        .collect()
}

fn calls_by_id(messages: &[Message]) -> HashMap<String, ToolCall> {
    messages
        .iter()
        .flat_map(|m| m.tool_calls.iter())
        .map(|c| (c.id.clone(), c.clone()))
        .collect()
}

/// The stub that replaces a tool result, or `None` when `m` is not a tool
/// result (or the stub would not be shorter). `seen`: an earlier request
/// showed the result in full.
fn stub_for(m: &Message, calls: &HashMap<String, ToolCall>, seen: bool) -> Option<String> {
    if m.role != Role::Tool {
        return None;
    }
    let (tool, args) = match m.tool_call_id.as_ref().and_then(|id| calls.get(id)) {
        Some(c) => (
            c.name.as_str(),
            clip(&c.arguments.to_string(), STUB_ARGS_CHARS),
        ),
        None => ("unknown", String::new()),
    };
    let chars = m.content.chars().count();
    let mut lines = m.content.lines().map(str::trim).filter(|l| !l.is_empty());
    let first_last = match (lines.next(), lines.next_back()) {
        (Some(first), Some(last)) => {
            format!(
                " (first line: {}; last line: {})",
                quote(first),
                quote(last)
            )
        }
        (Some(only), None) => format!(" (output: {})", quote(only)),
        _ => String::new(),
    };
    let stub = if seen {
        format!(
            "[earlier tool result elided to fit the context window: {tool} {args} returned \
             {chars} chars{first_last}. You read it in full when it ran; call the tool again only \
             if you need its exact content.]"
        )
    } else {
        format!(
            "[tool result not shown, to fit the context window: {tool} {args} returned {chars} \
             chars{first_last}. You have not seen its content. To read it, call the tool again, \
             with fewer calls in the same turn.]"
        )
    };
    (stub.chars().count() < message_chars(m)).then_some(stub)
}

/// Replace a tool result with its stub. Returns true if it changed.
fn apply_stub(m: &mut Message, calls: &HashMap<String, ToolCall>, seen: bool) -> bool {
    match stub_for(m, calls, seen) {
        Some(stub) => {
            m.content = stub;
            m.images.clear();
            m.audio.clear();
            true
        }
        None => false,
    }
}

fn quote(line: &str) -> String {
    format!("\"{}\"", clip(line, STUB_LINE_CHARS))
}

fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max).collect();
    out.push_str("...");
    out
}

#[async_trait]
impl Hook for ContextWindow {
    async fn before_model(&self, _ctx: &HookContext<'_>, request: &mut ModelRequest) {
        self.compact_request(request);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn body(len: usize, next: &str) -> String {
        format!("header line\n{}\nNext file: {next}\n", "x".repeat(len))
    }

    fn transcript(rounds: usize, result_len: usize) -> Vec<Message> {
        let mut m = vec![
            Message::system("sys"),
            Message::user("Summarize the files for Alice Rivera"),
        ];
        for r in 0..rounds {
            m.extend(round(
                &format!("c{r}"),
                &body(result_len, &format!("f{}", r + 1)),
            ));
        }
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

    fn view(window: &ContextWindow, m: &[Message]) -> Vec<Message> {
        let mut v = m.to_vec();
        window.compact(&mut v);
        v
    }

    /// Number of turns whose view does not extend the previous turn's.
    fn prefix_changes(window: &ContextWindow, turns: usize, result_len: usize) -> usize {
        let mut previous: Option<Vec<Message>> = None;
        let mut changes = 0;
        for r in 1..=turns {
            let v = view(window, &transcript(r, result_len));
            assert!(
                estimate_tokens(&v) <= window.budget_tokens,
                "turn {r}: {}",
                estimate_tokens(&v)
            );
            assert!(well_formed(&v));
            assert_eq!(v[1].content, "Summarize the files for Alice Rivera");
            if let Some(prev) = &previous
                && (v.len() < prev.len() || v[..prev.len()] != prev[..])
            {
                changes += 1;
            }
            previous = Some(v);
        }
        changes
    }

    #[test]
    fn under_budget_is_untouched() {
        let mut m = transcript(3, 100);
        let before = m.clone();
        assert!(!ContextWindow::new(10_000).compact(&mut m));
        assert_eq!(m, before);
    }

    #[test]
    fn old_results_become_stubs_that_say_what_they_were() {
        let mut m = transcript(10, 4_000);
        let window = ContextWindow::new(6_000);
        assert!(window.compact(&mut m));
        assert!(estimate_tokens(&m) <= 6_000);
        assert_eq!(m.len(), transcript(10, 4_000).len(), "nothing dropped");
        let stub = &m[3].content;
        assert!(stub.starts_with("[earlier tool result elided"), "{stub}");
        assert!(
            stub.contains(r#"read_file {"path":"c0"} returned 4027 chars"#),
            "{stub}"
        );
        assert!(stub.contains(r#"first line: "header line""#), "{stub}");
        assert!(stub.contains(r#"last line: "Next file: f1""#), "{stub}");
        // The newest turn is intact.
        let n = m.len();
        assert!(m[n - 1].content.len() > 4_000);
        assert!(well_formed(&m));
    }

    #[test]
    fn the_prefix_changes_only_in_coarse_steps() {
        // One ~1k-token read per turn under a 6k budget. Re-cutting every
        // turn would change the prefix on ~35 of 40 turns.
        let changes = prefix_changes(&ContextWindow::new(6_000), 40, 4_000);
        assert!(changes > 0);
        assert!(changes <= 40 / 3, "{changes} prefix changes in 40 turns");
    }

    #[test]
    fn stubs_are_dropped_in_steps_too() {
        // 200 turns: the stubs themselves outgrow the budget.
        let changes = prefix_changes(&ContextWindow::new(1_500), 200, 1_000);
        assert!(changes <= 200 / 3, "{changes} prefix changes in 200 turns");
    }

    #[test]
    fn recent_turns_over_budget_are_cut_for_that_request() {
        // One turn re-reads five big files at once: the protected tail
        // alone is over budget.
        let mut m = vec![Message::system("sys"), Message::user("go")];
        let calls: Vec<ToolCall> = (0..5)
            .map(|i| ToolCall::new(format!("c{i}"), "read_file", json!({"path": i})))
            .collect();
        m.push(Message::assistant("", calls));
        for i in 0..5 {
            m.push(Message::tool_result(format!("c{i}"), body(8_000, "z")));
        }
        let window = ContextWindow::new(5_000);
        assert!(window.compact(&mut m));
        assert!(estimate_tokens(&m) <= 5_000, "{}", estimate_tokens(&m));
        assert!(well_formed(&m));
        // The model never saw the elided results: the stub says so, and how
        // to get them.
        assert!(
            m[3].content.starts_with("[tool result not shown"),
            "{}",
            m[3].content
        );
        assert!(m[3].content.contains("You have not seen its content"));
        assert!(m[7].content.len() > 8_000, "the newest result stays");
    }

    /// A turn whose results did not all fit, then the model calls again for
    /// the ones it was not shown, alone or in a smaller batch: the repeated
    /// call comes back whole, and the first turn's stubs stay honest (the
    /// shown results say they were read, the others that they were not).
    #[test]
    fn a_result_never_shown_comes_back_whole_when_called_again() {
        let read = |id: &str, path: usize| ToolCall::new(id, "read_file", json!({"path": path}));
        let mut m = vec![Message::system("sys"), Message::user("sum the ten shards")];
        m.push(Message::assistant(
            "",
            (0..10).map(|i| read(&format!("a{i}"), i)).collect(),
        ));
        for i in 0..10 {
            m.push(Message::tool_result(
                format!("a{i}"),
                body(7_600, &format!("count {i}")),
            ));
        }
        let window = ContextWindow::new(8_000);
        let first = view(&window, &m);
        let unshown: Vec<usize> = (0..10)
            .filter(|i| first[3 + i].content.starts_with("[tool result not shown"))
            .collect();
        assert!(!unshown.is_empty() && unshown.len() < 10, "{unshown:?}");
        assert_eq!(unshown[0], 0, "oldest first");

        // Each unshown result is asked for again, one call per turn.
        for (turn, &i) in unshown.iter().enumerate() {
            let id = format!("r{turn}");
            m.push(Message::assistant("", vec![read(&id, i)]));
            m.push(Message::tool_result(id, body(7_600, &format!("count {i}"))));
            let v = view(&window, &m);
            assert!(estimate_tokens(&v) <= 8_000);
            assert!(well_formed(&v));
            let last = v.last().unwrap();
            assert!(last.content.len() > 7_600, "re-read {i} came back stubbed");
            assert!(last.content.contains(&format!("count {i}")));
            for (j, msg) in v.iter().enumerate() {
                let Some(id) = msg.tool_call_id.as_deref() else {
                    continue;
                };
                let Some(k) = id.strip_prefix('a').and_then(|k| k.parse::<usize>().ok()) else {
                    continue;
                };
                if msg.content.starts_with('[') {
                    let honest = if unshown.contains(&k) {
                        "[tool result not shown"
                    } else {
                        "[earlier tool result elided"
                    };
                    assert!(
                        msg.content.starts_with(honest),
                        "turn {turn}, {j}: {}",
                        msg.content
                    );
                }
            }
        }
    }

    #[test]
    fn tool_definitions_count_against_the_budget() {
        let mut request = ModelRequest {
            model: "m".into(),
            messages: transcript(4, 4_000),
            tools: vec![aidaemon_core::ToolSpec {
                name: "read_file".into(),
                description: "d".repeat(8_000),
                parameters: json!({}),
            }],
            tool_choice: Default::default(),
            max_output_tokens: None,
            options: Default::default(),
        };
        assert!(estimate_tokens(&request.messages) < 6_000);
        assert!(ContextWindow::new(6_000).compact_request(&mut request));
        assert!(estimate_tokens(&request.messages) + 2_000 <= 6_000);
    }

    #[test]
    fn the_latest_user_message_is_never_dropped() {
        // A second question, then big tool rounds: the question the model
        // is answering must survive both the stable steps and the
        // emergency drop.
        let mut m = vec![
            Message::system("sys"),
            Message::user("Summarize the files for Alice Rivera"),
            Message::assistant("Done: three files summarized.", Vec::new()),
            Message::user("Now which one mentions the refund?"),
        ];
        for r in 0..6 {
            m.extend(round(&format!("c{r}"), &body(8_000, "z")));
        }
        for budget in [3_000, 6_000, 12_000] {
            let v = view(&ContextWindow::new(budget), &m);
            assert!(well_formed(&v));
            assert!(estimate_tokens(&v) <= budget, "{}", estimate_tokens(&v));
            assert!(
                v.iter()
                    .any(|x| x.content == "Now which one mentions the refund?"),
                "budget {budget}: the current question was dropped"
            );
            assert_eq!(v[0].content, "sys");
        }
    }

    #[test]
    fn protected_part_over_budget_is_sent_as_is() {
        let mut m = vec![Message::system("s".repeat(10_000)), Message::user("hi")];
        let before = m.clone();
        ContextWindow::new(10).compact(&mut m);
        assert_eq!(m, before);
    }
}
