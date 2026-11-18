//! Compact encoding of approval answers into Telegram `callback_data`.
//!
//! Telegram limits `callback_data` to 64 bytes. When the prompt id and
//! choice id fit, they are encoded inline and survive restarts:
//! `a1:<prompt_len>:<prompt_id><choice_id>`. Otherwise the pair is kept in a
//! bounded in-memory table and the button carries a short key: `a2:<n>`.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

/// Telegram's maximum `callback_data` length, in bytes.
pub const MAX_CALLBACK_DATA: usize = 64;

const INLINE_PREFIX: &str = "a1:";
const KEYED_PREFIX: &str = "a2:";
const MAX_KEYED: usize = 1024;

#[derive(Default)]
pub(crate) struct CallbackCodec {
    inner: Mutex<Keyed>,
}

#[derive(Default)]
struct Keyed {
    next: u64,
    map: HashMap<u64, (String, String)>,
    order: VecDeque<u64>,
}

impl CallbackCodec {
    pub(crate) fn encode(&self, prompt_id: &str, choice_id: &str) -> String {
        let inline = format!("{INLINE_PREFIX}{}:{prompt_id}{choice_id}", prompt_id.len());
        if inline.len() <= MAX_CALLBACK_DATA {
            return inline;
        }
        let mut keyed = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let key = keyed.next;
        keyed.next += 1;
        keyed
            .map
            .insert(key, (prompt_id.to_string(), choice_id.to_string()));
        keyed.order.push_back(key);
        while keyed.order.len() > MAX_KEYED {
            if let Some(old) = keyed.order.pop_front() {
                keyed.map.remove(&old);
            }
        }
        format!("{KEYED_PREFIX}{key}")
    }

    /// Returns `(prompt_id, choice_id)`, or `None` for data this adapter did
    /// not produce (or a keyed entry that has expired).
    pub(crate) fn decode(&self, data: &str) -> Option<(String, String)> {
        if let Some(rest) = data.strip_prefix(INLINE_PREFIX) {
            let (len, rest) = rest.split_once(':')?;
            let len: usize = len.parse().ok()?;
            if !rest.is_char_boundary(len) || len > rest.len() {
                return None;
            }
            let (prompt, choice) = rest.split_at(len);
            return Some((prompt.to_string(), choice.to_string()));
        }
        if let Some(rest) = data.strip_prefix(KEYED_PREFIX) {
            let key: u64 = rest.parse().ok()?;
            let keyed = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            return keyed.map.get(&key).cloned();
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_roundtrip() {
        let codec = CallbackCodec::default();
        let data = codec.encode("p:1|x", "allow_once");
        assert!(data.len() <= MAX_CALLBACK_DATA);
        assert_eq!(
            codec.decode(&data),
            Some(("p:1|x".to_string(), "allow_once".to_string()))
        );
    }

    #[test]
    fn long_ids_use_keyed_table() {
        let codec = CallbackCodec::default();
        let prompt = "prompt-".repeat(20);
        let data = codec.encode(&prompt, "deny");
        assert!(data.starts_with(KEYED_PREFIX));
        assert!(data.len() <= MAX_CALLBACK_DATA);
        assert_eq!(codec.decode(&data), Some((prompt, "deny".to_string())));
    }

    #[test]
    fn foreign_data_is_rejected() {
        let codec = CallbackCodec::default();
        assert_eq!(codec.decode("something_else"), None);
        assert_eq!(codec.decode("a1:99:short"), None);
        assert_eq!(codec.decode("a2:12345"), None);
    }
}
