//! Mandate review capacity: the owner-facing effort levels, the token budget
//! a review cadence needs, and whether a controller can fund its next
//! review today. Owner tools and listings share it.

use serde_json::{Value, json};

/// The owner-facing review effort, which sets a mandate's token capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReviewEffort {
    Efficient,
    Balanced,
    Thorough,
}

impl ReviewEffort {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw {
            "efficient" => Some(Self::Efficient),
            "balanced" => Some(Self::Balanced),
            "thorough" => Some(Self::Thorough),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Efficient => "efficient",
            Self::Balanced => "balanced",
            Self::Thorough => "thorough",
        }
    }

    pub const fn per_cycle_capacity(self) -> i64 {
        match self {
            Self::Efficient => 100_000,
            Self::Balanced => 250_000,
            Self::Thorough => 500_000,
        }
    }

    pub const fn daily_capacity_floor(self) -> i64 {
        match self {
            Self::Efficient => 1_000_000,
            Self::Balanced => 2_000_000,
            Self::Thorough => 4_000_000,
        }
    }
}

/// How many reviews at the default interval fit the mandate's first day
/// (or its shorter lifetime: `duration_secs`, else `expires_at`).
pub fn expected_default_review_cycles(
    default_review_secs: i64,
    duration_secs: Option<i64>,
    expires_at: Option<&str>,
) -> anyhow::Result<i64> {
    anyhow::ensure!(
        default_review_secs > 0,
        "default review interval must be positive"
    );
    let horizon_secs = if let Some(duration_secs) = duration_secs {
        duration_secs
    } else if let Some(expires_at) = expires_at {
        let expires_at = chrono::DateTime::parse_from_rfc3339(expires_at)
            .map_err(|_| anyhow::anyhow!("expires_at must be an RFC3339 timestamp"))?
            .with_timezone(&chrono::Utc);
        (expires_at - chrono::Utc::now()).num_seconds().max(1)
    } else {
        24 * 60 * 60
    }
    .clamp(1, 24 * 60 * 60);

    // The first review is immediate, then subsequent reviews occur at the
    // default interval. Exclude a cycle exactly at expiration.
    Ok(((horizon_secs - 1) / default_review_secs) + 1)
}

/// The controller goal's token use today against its daily budget, as the
/// owner-facing policy view shows it.
pub fn controller_budget_snapshot(goal: &crate::traits::Goal, review_effort: &str) -> Value {
    let now = chrono::Utc::now();
    let today = now.date_naive().to_string();
    let used_today = if goal.tokens_used_day == today {
        goal.tokens_used_today.max(0)
    } else {
        0
    };
    let remaining = goal
        .budget_daily
        .map(|daily| daily.saturating_sub(used_today).max(0));
    let can_fund_full_cycle = match (remaining, goal.budget_per_check) {
        (Some(remaining), Some(per_cycle)) => remaining >= per_cycle.max(0),
        _ => true,
    };
    let blocked_until_utc = (!can_fund_full_cycle).then(|| {
        let midnight = now
            .date_naive()
            .succ_opt()
            .expect("the next UTC date is representable")
            .and_hms_opt(0, 0, 0)
            .expect("UTC midnight is representable");
        chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(midnight, chrono::Utc)
            .to_rfc3339()
    });
    let used_percent = goal
        .budget_daily
        .filter(|daily| *daily > 0)
        .map(|daily| ((used_today as f64 / daily as f64) * 100.0).clamp(0.0, 100.0));
    let effort = ReviewEffort::parse(review_effort);
    json!({
        "review_effort": effort.map_or("legacy_custom", ReviewEffort::as_str),
        "automatically_managed": effort.is_some(),
        "runaway_protection": true,
        "usage_percent_today": used_percent,
        "usage_utc_day": today,
        "can_fund_full_cycle": can_fund_full_cycle,
        "blocked_until_utc": blocked_until_utc,
    })
}

/// `ready`, or when the next review can be funded.
pub fn controller_budget_admission_label(
    goal: &crate::traits::Goal,
    review_effort: &str,
) -> String {
    let snapshot = controller_budget_snapshot(goal, review_effort);
    if snapshot["can_fund_full_cycle"] == Value::Bool(true) {
        "ready".to_string()
    } else {
        format!(
            "budget-blocked-until {}",
            snapshot["blocked_until_utc"]
                .as_str()
                .unwrap_or("next UTC reset")
        )
    }
}
