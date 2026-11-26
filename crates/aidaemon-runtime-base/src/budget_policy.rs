//! Token budget limits and the pure policy for a used-up budget, below the
//! engine so authority and the turn can name them.

pub const GOAL_BUDGET_EXHAUSTED: &str = "the goal's token budget is used up";
/// How often, and how far, a used-up budget may extend.
pub const TASK_MAX_BUDGET_EXTENSIONS: usize = 3;
pub const TASK_HARD_TOKEN_CAP: u64 = 2_000_000;
pub const SCHEDULED_MAX_BUDGET_EXTENSIONS: usize = 1;
pub const SCHEDULED_HARD_TOKEN_CAP: i64 = 2_000_000;
pub const MANDATE_BUDGET_EXHAUSTED: &str = "the mandate cycle's token balance is exhausted";

/// The task budget's one 80% warning is due: past 80%, not yet exhausted,
/// not already given.
pub fn task_budget_warning_due(used: u64, budget: u64, already_sent: bool) -> bool {
    budget > 0 && !already_sent && used >= budget.saturating_mul(80) / 100 && used < budget
}

/// The run's task token budget (0 means none). A scheduled run with a
/// per-check budget is bounded by that run budget (with its own extension
/// and alert), so the task budget does not stack on top of it. Without one,
/// the configured task budget applies.
pub fn task_token_budget(configured: Option<u64>, scheduled_per_check: Option<i64>) -> u64 {
    match scheduled_per_check.filter(|budget| *budget > 0) {
        Some(_) => 0,
        None => configured.unwrap_or(0),
    }
}

/// One request's task token budget: the whole tree's, the turn and every
/// plain sub-agent it starts. Its tokens, its limit, its extensions and its
/// progress are one record, so the tree is charged once and extended once
/// (live 2026-11-13: two helpers charged the tree's shared counter but each
/// held its own 500,000 limit and its own extension count, so each
/// compared the tree's 513,391 tokens with its own limit, one extended on
/// its own and the other asked the owner mid-run).
#[derive(Debug, Default)]
pub struct TreeBudget {
    tokens: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// 0: none. Set by the tree's first turn ([`Self::start`]).
    limit: std::sync::atomic::AtomicU64,
    started: std::sync::atomic::AtomicBool,
    /// Distinct successful operations across the tree: its progress.
    successes: std::sync::atomic::AtomicUsize,
    /// The same, per member (the turn and each helper, by session): what a
    /// member that is repeating itself compares with the rest of the tree.
    by_member: std::sync::Mutex<std::collections::HashMap<String, usize>>,
    /// Each member's count at the last extension.
    by_member_at_last: std::sync::Mutex<std::collections::HashMap<String, usize>>,
    warned: std::sync::atomic::AtomicBool,
    /// Extensions so far, and the progress at the last one. Held while one
    /// member decides, so a second member that hit the same limit waits for
    /// that decision instead of making its own.
    extensions: tokio::sync::Mutex<Extensions>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Extensions {
    pub count: usize,
    pub successes_at_last: usize,
}

impl TreeBudget {
    /// The tree's budget as its first turn sets it; later members keep it.
    pub fn start(&self, limit: u64) {
        if !self.started.swap(true, std::sync::atomic::Ordering::SeqCst) {
            self.limit.store(limit, std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// The tree's token counter.
    pub fn tokens(&self) -> std::sync::Arc<std::sync::atomic::AtomicU64> {
        self.tokens.clone()
    }

    /// Charge `delta` tokens; the tree's total after it.
    pub fn charge(&self, delta: u64) -> u64 {
        self.tokens
            .fetch_add(delta, std::sync::atomic::Ordering::Relaxed)
            + delta
    }

    pub fn used(&self) -> u64 {
        self.tokens.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn limit(&self) -> u64 {
        self.limit.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Member `member` made a new distinct successful operation.
    pub fn note_success(&self, member: &str) {
        self.successes
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Ok(mut by_member) = self.by_member.lock() {
            *by_member.entry(member.to_string()).or_default() += 1;
        }
    }

    /// Whether the tree is making progress for `member`, whose latest call
    /// repeated an earlier one when `repeating`. The budget is the tree's,
    /// so its progress is the tree's: something new succeeded anywhere in it
    /// since the last extension. Only a member going round in circles needs
    /// the rest of the tree to have done new work since then: a helper
    /// checking on its running command while its sibling works is waiting,
    /// not stuck, whichever of them did the very latest new thing (live
    /// 2026-11-14: asked at 500,044 of 500,000 tokens, "Nothing new has
    /// succeeded", while two helpers were installing a site each).
    pub fn progressed(&self, extensions: &Extensions, member: &str, repeating: bool) -> bool {
        let new_in_tree = self
            .successes()
            .saturating_sub(extensions.successes_at_last);
        if !repeating {
            return new_in_tree > 0;
        }
        let count = |map: &std::sync::Mutex<std::collections::HashMap<String, usize>>| {
            map.lock()
                .ok()
                .and_then(|map| map.get(member).copied())
                .unwrap_or(0)
        };
        let own_new = count(&self.by_member).saturating_sub(count(&self.by_member_at_last));
        new_in_tree > own_new
    }

    pub fn successes(&self) -> usize {
        self.successes.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The tree's one 80% warning: `true` once, for the member that gives it.
    pub fn take_warning(&self) -> bool {
        let (used, limit) = (self.used(), self.limit());
        task_budget_warning_due(
            used,
            limit,
            self.warned.load(std::sync::atomic::Ordering::Relaxed),
        ) && !self.warned.swap(true, std::sync::atomic::Ordering::Relaxed)
    }

    /// Decide over the tree's extensions, one member at a time.
    pub async fn deciding(&self) -> tokio::sync::MutexGuard<'_, Extensions> {
        self.extensions.lock().await
    }

    /// Raise the limit, recording the extension and the progress at it.
    pub fn extend(&self, to: u64, extensions: &mut Extensions) {
        self.limit.store(to, std::sync::atomic::Ordering::SeqCst);
        extensions.count += 1;
        extensions.successes_at_last = self.successes();
        if let (Ok(now), Ok(mut at_last)) = (self.by_member.lock(), self.by_member_at_last.lock()) {
            *at_last = now.clone();
        }
    }
}

/// What a used-up task budget does next, under the configured
/// auto-extension policy: up to [`TASK_MAX_BUDGET_EXTENSIONS`] extensions,
/// never past [`TASK_HARD_TOKEN_CAP`], each while the tree made new
/// progress since the last.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetStep {
    /// Within the policy: extend without asking.
    Extend { to: u64 },
    /// Past the policy: more spending is the owner's decision.
    Ask { to: u64, why: AskWhy },
    /// At the hard cap: nothing can extend it.
    Stop,
}

/// Why spending more is the owner's decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AskWhy {
    /// The automatic extensions the policy allows are used up.
    ExtensionsUsed(usize),
    /// Nothing new succeeded anywhere in the request since the budget last
    /// grew.
    NoProgress,
    /// The request is not the owner's own.
    NotOwners,
}

impl AskWhy {
    /// Why the owner is asked, in plain words.
    pub fn words(self) -> String {
        match self {
            Self::ExtensionsUsed(count) => format!(
                "It already grew {count} time{} on its own, the most the budget policy allows.",
                if count == 1 { "" } else { "s" }
            ),
            Self::NoProgress => {
                "Nothing new has succeeded since the budget last grew, so it may be stuck."
                    .to_string()
            }
            Self::NotOwners => "This request did not come from you.".to_string(),
        }
    }
}

/// The policy for a used-up task budget of `limit` at `used` tokens, given
/// whether the tree made progress since its last extension
/// ([`TreeBudget::progressed`]).
pub fn task_budget_step(
    used: u64,
    limit: u64,
    extensions: Extensions,
    progressed: bool,
    owners: bool,
) -> BudgetStep {
    let cap = TASK_HARD_TOKEN_CAP;
    let to = limit
        .saturating_mul(2)
        .max(used.saturating_add(limit / 2))
        .min(cap);
    if limit >= cap || to <= used {
        return BudgetStep::Stop;
    }
    let why = if !owners {
        AskWhy::NotOwners
    } else if extensions.count >= TASK_MAX_BUDGET_EXTENSIONS {
        AskWhy::ExtensionsUsed(extensions.count)
    } else if !progressed {
        AskWhy::NoProgress
    } else {
        return BudgetStep::Extend { to };
    };
    BudgetStep::Ask { to, why }
}
