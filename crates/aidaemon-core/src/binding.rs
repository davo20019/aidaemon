//! Binding an answer to the question it answers.
//!
//! An interruption carries a keyed digest ([`crate::Interruption::digest`])
//! of exactly what was asked: the run, the turn, the call (id, tool and
//! canonical arguments), the hook that asked and the hooks the owner already
//! answered for this call. `Resolution::Approve` waives those hooks only when
//! the saved call still matches the digest, and only once: a copy of the
//! state resumed again is refused ([`crate::CoreError::ApprovalAlreadyUsed`]).
//! A call that does not match (edited, swapped, or saved before digests
//! existed) is judged again from the first hook, so the owner is asked about
//! what would really run.
//!
//! The key is random per process unless the application sets one
//! ([`crate::AgentBuilder::approval_key`]), so an approval given in one
//! process and resumed in another is asked again unless both share a key.

use std::collections::{HashSet, VecDeque};
use std::sync::{Mutex, OnceLock};

use hmac::{KeyInit, Mac};
use serde_json::Value;

use crate::message::ToolCall;
use crate::state::RunId;

type HmacSha256 = hmac::Hmac<sha2::Sha256>;

/// How many used approvals the process remembers.
const USED_CAPACITY: usize = 65_536;

/// The process's own key, when the application gave none.
pub(crate) fn process_key() -> [u8; 32] {
    static KEY: OnceLock<[u8; 32]> = OnceLock::new();
    *KEY.get_or_init(|| {
        let mut key = [0u8; 32];
        key[..16].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
        key[16..].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
        key
    })
}

/// What one question is about.
pub(crate) struct Question<'a> {
    pub run_id: RunId,
    pub turn: u32,
    pub call: &'a ToolCall,
    pub hook_index: usize,
    pub hook_name: &'a str,
    /// Hooks already answered for this call, in ascending order.
    pub waived: &'a [usize],
}

/// The keyed digest of `question`, as lowercase hex.
pub(crate) fn digest(key: &[u8; 32], question: &Question<'_>) -> String {
    let material = serde_json::json!([
        question.run_id.to_string(),
        question.turn,
        question.call.id,
        question.call.name,
        canonical(&question.call.arguments),
        question.hook_index,
        question.hook_name,
        question.waived,
    ]);
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC takes any key length");
    mac.update(material.to_string().as_bytes());
    hex::encode(mac.finalize().into_bytes())
}

/// Constant-time comparison of two digests.
pub(crate) fn same(a: &str, b: &str) -> bool {
    a.len() == b.len()
        && a.bytes()
            .zip(b.bytes())
            .fold(0u8, |acc, (x, y)| acc | (x ^ y))
            == 0
}

/// `value` with every object's keys sorted, so a digest does not depend on
/// the order a map kept its keys in.
fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut out = serde_json::Map::new();
            for key in keys {
                out.insert(key.clone(), canonical(&map[key]));
            }
            // Insertion order is the sorted order whatever the map type.
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
        other => other.clone(),
    }
}

/// Record `digest` as used. Returns `false` when it already was.
pub(crate) fn consume(digest: &str) -> bool {
    static USED: OnceLock<Mutex<(HashSet<String>, VecDeque<String>)>> = OnceLock::new();
    let used = USED.get_or_init(Mutex::default);
    let mut used = used.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let (set, order) = &mut *used;
    if !set.insert(digest.to_string()) {
        return false;
    }
    order.push_back(digest.to_string());
    if order.len() > USED_CAPACITY
        && let Some(oldest) = order.pop_front()
    {
        set.remove(&oldest);
    }
    true
}
