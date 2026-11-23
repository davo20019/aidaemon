//! Working-directory claims and prompt deduplication between runs.

use super::*;

/// A working directory claim with enough metadata to explain conflicts.
pub(super) struct WorkingDirClaim {
    pub(super) task_id: String,
    pub(super) tool_name: String,
    pub(super) prompt_summary: String,
    pub(super) dedup_prompt: String,
}

/// The claims map uses a std (not tokio) Mutex so that WorkingDirClaimGuard
/// can release a claim in Drop (Drop cannot await). Never hold this lock
/// across an await point; the compiler enforces it for Send futures.
pub(super) type WorkingDirClaims = std::sync::Mutex<HashMap<String, WorkingDirClaim>>;

/// Lock the claims map, recovering from poisoning (a panic while holding the
/// lock must not permanently wedge working-dir dispatch).
pub(super) fn lock_claims(
    claims: &WorkingDirClaims,
) -> std::sync::MutexGuard<'_, HashMap<String, WorkingDirClaim>> {
    claims
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// RAII guard that releases a working-dir claim on drop unless disarmed.
///
/// Dispatch futures can be dropped at any await point when the calling
/// session is cancelled or times out. Before this guard existed, an abort
/// between claim insertion and the explicit post-await release leaked the
/// claim permanently, blocking all future dispatches to that directory
/// until a daemon restart. The guard makes release abort-safe.
///
/// Disarm when handing the task off to the background `running` map: from
/// that point the reaper / cancel paths own the release.
pub(super) struct WorkingDirClaimGuard {
    pub(super) claims: Arc<WorkingDirClaims>,
    pub(super) dir: String,
    pub(super) task_id: String,
    pub(super) armed: bool,
}

impl WorkingDirClaimGuard {
    pub(super) fn new(claims: Arc<WorkingDirClaims>, dir: String, task_id: String) -> Self {
        Self {
            claims,
            dir,
            task_id,
            armed: true,
        }
    }

    /// Stop the guard from releasing on drop (ownership of the claim has
    /// been transferred to the background `running` map).
    pub(super) fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for WorkingDirClaimGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut claims = lock_claims(&self.claims);
        let owned = claims
            .get(&self.dir)
            .map(|claim| claim.task_id == self.task_id)
            .unwrap_or(false);
        if owned {
            claims.remove(&self.dir);
        }
    }
}

/// Jaccard similarity between two strings based on word-level bigrams.
/// Returns a value in [0.0, 1.0].
pub(super) fn prompt_similarity(a: &str, b: &str) -> f64 {
    fn bigrams(s: &str) -> HashSet<(String, String)> {
        let words: Vec<&str> = s.split_whitespace().collect();
        if words.len() < 2 {
            // For single-word prompts, fall back to unigram set
            return words
                .into_iter()
                .map(|w| (w.to_string(), String::new()))
                .collect();
        }
        words
            .windows(2)
            .map(|w| (w[0].to_string(), w[1].to_string()))
            .collect()
    }

    let set_a = bigrams(&a.to_lowercase());
    let set_b = bigrams(&b.to_lowercase());

    if set_a.is_empty() && set_b.is_empty() {
        return 1.0; // Both empty → identical
    }
    if set_a.is_empty() || set_b.is_empty() {
        return 0.0;
    }

    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Build a dedup key from a prompt: first 200 chars, lowercased.
pub(super) fn make_dedup_prompt(prompt: &str) -> String {
    prompt.chars().take(200).collect::<String>().to_lowercase()
}
