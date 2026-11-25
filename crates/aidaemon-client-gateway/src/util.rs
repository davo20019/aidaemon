//! Small helpers the gateway's modules share.

use std::sync::{Mutex, MutexGuard};

use sha2::{Digest, Sha256};

/// Lowercase hex SHA-256 of `text` (how tokens, tickets and keys are kept).
pub(crate) fn sha(text: &str) -> String {
    hex::encode(Sha256::digest(text.as_bytes()))
}

/// The guard, even after a panic poisoned the lock.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}
