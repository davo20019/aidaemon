use serde::{Deserialize, Serialize};

use crate::receipt::{Receipt, ReceiptStatus};

/// Thresholds for loop detection. Configurable on purpose: OpenHands
/// hardcoded theirs and legitimate polling got flagged (their issue #762).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StuckConfig {
    /// Identical call with identical output this many times in a row.
    pub repeat_threshold: usize,
    /// Same tool failing this many times in a row (arguments may differ).
    pub error_threshold: usize,
    /// Strict A, B, A, B alternation over this many calls. 0 disables.
    pub ping_pong_window: usize,
    /// With the default policy, stop after this many nudges.
    pub max_nudges: u32,
}

impl Default for StuckConfig {
    fn default() -> Self {
        Self {
            repeat_threshold: 3,
            error_threshold: 3,
            ping_pong_window: 6,
            max_nudges: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum StuckSignal {
    RepeatedCall { tool: String, count: usize },
    RepeatedError { tool: String, count: usize },
    PingPong { first: String, second: String },
}

impl StuckSignal {
    pub(crate) fn nudge_text(&self) -> String {
        match self {
            Self::RepeatedCall { tool, count } => format!(
                "[harness] You have called `{tool}` {count} times with the same arguments and \
                 received the same result. Repeating it will not change the outcome. Try a \
                 different approach, or explain what is blocking you."
            ),
            Self::RepeatedError { tool, count } => format!(
                "[harness] `{tool}` has failed {count} times in a row. Read the last error \
                 carefully and change approach instead of retrying the same way."
            ),
            Self::PingPong { first, second } => format!(
                "[harness] You are alternating between `{first}` and `{second}` without making \
                 progress. Step back and pick a different approach."
            ),
        }
    }
}

/// One observed call. Fingerprints use FNV-1a so they are stable across
/// processes and survive serialization in [`crate::RunState`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Observation {
    tool: String,
    call_fp: u64,
    pair_fp: u64,
    failed: bool,
}

/// Rolling window over recent calls.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StuckState {
    history: Vec<Observation>,
}

const WINDOW: usize = 32;

impl StuckState {
    pub(crate) fn observe(
        &mut self,
        config: &StuckConfig,
        receipt: &Receipt,
        output: &str,
    ) -> Option<StuckSignal> {
        let args = receipt.arguments.to_string();
        let call_fp = fnv1a(&[receipt.tool.as_bytes(), b"\0", args.as_bytes()]);
        let pair_fp = fnv1a(&[&call_fp.to_le_bytes(), output.as_bytes()]);
        let failed = !matches!(receipt.status, ReceiptStatus::Succeeded);
        self.history.push(Observation {
            tool: receipt.tool.clone(),
            call_fp,
            pair_fp,
            failed,
        });
        if self.history.len() > WINDOW {
            self.history.remove(0);
        }
        self.detect(config)
    }

    pub(crate) fn reset(&mut self) {
        self.history.clear();
    }

    fn detect(&self, config: &StuckConfig) -> Option<StuckSignal> {
        let h = &self.history;
        let last = h.last()?;

        let n = config.repeat_threshold;
        if n > 1 && h.len() >= n && h[h.len() - n..].iter().all(|o| o.pair_fp == last.pair_fp) {
            return Some(StuckSignal::RepeatedCall {
                tool: last.tool.clone(),
                count: n,
            });
        }

        let n = config.error_threshold;
        if n > 1
            && h.len() >= n
            && h[h.len() - n..]
                .iter()
                .all(|o| o.failed && o.tool == last.tool)
        {
            return Some(StuckSignal::RepeatedError {
                tool: last.tool.clone(),
                count: n,
            });
        }

        let w = config.ping_pong_window;
        if w >= 4 && h.len() >= w {
            let tail = &h[h.len() - w..];
            let (a, b) = (tail[0].call_fp, tail[1].call_fp);
            let alternates = a != b
                && tail
                    .iter()
                    .enumerate()
                    .all(|(i, o)| o.call_fp == if i % 2 == 0 { a } else { b });
            if alternates {
                return Some(StuckSignal::PingPong {
                    first: tail[0].tool.clone(),
                    second: tail[1].tool.clone(),
                });
            }
        }
        None
    }
}

fn fnv1a(parts: &[&[u8]]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for part in parts {
        for byte in *part {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::receipt::InvocationStage;
    use serde_json::json;

    fn receipt(tool: &str, args: serde_json::Value, status: ReceiptStatus) -> Receipt {
        Receipt {
            run_id: crate::state::RunId::new(),
            call_id: "c".into(),
            tool: tool.into(),
            arguments: args,
            turn: 1,
            stage: InvocationStage::Dispatched,
            status,
            read_only: true,
            duration_ms: 0,
            reason: None,
            details: serde_json::Value::Null,
        }
    }

    #[test]
    fn repeated_identical_pairs_fire() {
        let cfg = StuckConfig::default();
        let mut s = StuckState::default();
        let r = receipt("ls", json!({"p": "."}), ReceiptStatus::Succeeded);
        assert!(s.observe(&cfg, &r, "a b").is_none());
        assert!(s.observe(&cfg, &r, "a b").is_none());
        assert!(matches!(
            s.observe(&cfg, &r, "a b"),
            Some(StuckSignal::RepeatedCall { .. })
        ));
    }

    #[test]
    fn same_call_with_changing_output_is_polling_not_stuck() {
        let cfg = StuckConfig::default();
        let mut s = StuckState::default();
        let r = receipt("status", json!({}), ReceiptStatus::Succeeded);
        for i in 0..10 {
            assert!(s.observe(&cfg, &r, &format!("progress {i}%")).is_none());
        }
    }

    #[test]
    fn repeated_errors_fire_even_with_different_args() {
        let cfg = StuckConfig::default();
        let mut s = StuckState::default();
        for i in 0..2 {
            let r = receipt("build", json!({"try": i}), ReceiptStatus::Failed);
            assert!(s.observe(&cfg, &r, &format!("err {i}")).is_none());
        }
        let r = receipt("build", json!({"try": 9}), ReceiptStatus::Failed);
        assert!(matches!(
            s.observe(&cfg, &r, "err 9"),
            Some(StuckSignal::RepeatedError { .. })
        ));
    }

    #[test]
    fn ping_pong_fires() {
        let cfg = StuckConfig::default();
        let mut s = StuckState::default();
        let mut last = None;
        for i in 0..6 {
            let (tool, out) = if i % 2 == 0 { ("a", i) } else { ("b", i) };
            let r = receipt(tool, json!({}), ReceiptStatus::Succeeded);
            last = s.observe(&cfg, &r, &out.to_string());
        }
        assert!(matches!(last, Some(StuckSignal::PingPong { .. })));
    }
}
