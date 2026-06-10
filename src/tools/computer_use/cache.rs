#![allow(dead_code)]

use std::collections::HashMap;

use super::types::{AppSnapshot, IndexedElement};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SnapshotKey {
    pub task_id: String,
    pub session_id: String,
    pub bundle_id: String,
}

#[derive(Debug, Clone)]
struct CachedSnapshot {
    snapshot: AppSnapshot,
    state_hash: u64,
}

#[derive(Debug, Default)]
pub struct SnapshotCache {
    generation_counter: u64,
    current: HashMap<SnapshotKey, CachedSnapshot>,
}

impl SnapshotCache {
    pub fn next_generation(&mut self) -> u64 {
        self.generation_counter = self.generation_counter.saturating_add(1);
        self.generation_counter
    }

    pub fn store(&mut self, key: SnapshotKey, mut snapshot: AppSnapshot) -> AppSnapshot {
        let generation = self.next_generation();
        snapshot.generation = generation;
        let state_hash = hash_snapshot(&snapshot);
        let cached = CachedSnapshot {
            snapshot: snapshot.clone(),
            state_hash,
        };
        self.current.insert(key, cached);
        snapshot
    }

    pub fn current_generation(&self, key: &SnapshotKey) -> Option<u64> {
        self.current.get(key).map(|c| c.snapshot.generation)
    }

    pub fn validate_generation(
        &self,
        key: &SnapshotKey,
        generation: u64,
    ) -> Result<&AppSnapshot, String> {
        let Some(cached) = self.current.get(key) else {
            return Err(
                "No snapshot for this app in the current task — call get_app_state first".into(),
            );
        };
        // Only reject generations that are *behind* the current snapshot — those
        // are genuinely stale (the UI may have changed since, so an element index
        // could now point at a different control). A generation that matches or
        // is *ahead* of current is bound to the freshest snapshot: a weaker model
        // that loses track of the monotonic counter and guesses a future number
        // would otherwise loop forever (reject → re-inspect → guess ahead again).
        // The current snapshot is the newest available, so validating against it
        // preserves the staleness guarantee (the index is checked against the
        // live tree) without the off-by-one livelock.
        if generation < cached.snapshot.generation {
            return Err(format!(
                "Stale snapshot_generation {generation} (current is {}). Call get_app_state again.",
                cached.snapshot.generation
            ));
        }
        Ok(&cached.snapshot)
    }

    pub fn element_by_index(
        &self,
        key: &SnapshotKey,
        generation: u64,
        index: u32,
    ) -> Result<&IndexedElement, String> {
        let snapshot = self.validate_generation(key, generation)?;
        snapshot
            .elements
            .iter()
            .find(|e| e.interactive && e.index == index)
            .ok_or_else(|| {
                format!(
                    "element_index {index} not found in snapshot generation {}",
                    snapshot.generation
                )
            })
    }

    pub fn clear_task(&mut self, task_id: &str) {
        self.current.retain(|k, _| k.task_id != task_id);
    }

    pub fn state_hash(&self, key: &SnapshotKey) -> Option<u64> {
        self.current.get(key).map(|c| c.state_hash)
    }
}

fn hash_snapshot(snapshot: &AppSnapshot) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    snapshot.bundle_id.hash(&mut hasher);
    snapshot.pid.hash(&mut hasher);
    snapshot.window_id.hash(&mut hasher);
    snapshot.window_title.hash(&mut hasher);
    snapshot.elements.len().hash(&mut hasher);
    for el in &snapshot.elements {
        el.index.hash(&mut hasher);
        el.role.hash(&mut hasher);
        el.title.hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::computer_use::types::IndexedElement;

    fn key() -> SnapshotKey {
        SnapshotKey {
            task_id: "t1".into(),
            session_id: "s1".into(),
            bundle_id: "com.apple.calculator".into(),
        }
    }

    fn sample_snapshot() -> AppSnapshot {
        AppSnapshot {
            generation: 0,
            bundle_id: "com.apple.calculator".into(),
            app_name: "Calculator".into(),
            pid: 10,
            window_id: 1,
            window_title: "Calculator".into(),
            elements: vec![IndexedElement {
                index: 1,
                role: "AXButton".into(),
                title: "7".into(),
                enabled: true,
                bounds: None,
                subrole: None,
                interactive: true,
            }],
            truncated: false,
            png: vec![1, 2, 3],
        }
    }

    #[test]
    fn stale_generation_is_rejected() {
        let mut cache = SnapshotCache::default();
        let snap = cache.store(key(), sample_snapshot());
        assert!(cache.validate_generation(&key(), snap.generation).is_ok());
        assert!(cache
            .validate_generation(&key(), snap.generation - 1)
            .is_err());
    }

    #[test]
    fn generations_monotonically_increase() {
        let mut cache = SnapshotCache::default();
        let a = cache.store(key(), sample_snapshot());
        let b = cache.store(key(), sample_snapshot());
        assert!(b.generation > a.generation);
    }
}
