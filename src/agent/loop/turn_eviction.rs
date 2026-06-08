//! Pillar B: whole-turn eviction against the archived-region budget. Spec §Eviction.
//!
//! The new field/module is unused until Task 7 (the message-build integration
//! consumes it); keep the lib clippy gate clean in the meantime.
#![allow(dead_code)]

pub(crate) fn archived_budget(
    context_budget: usize,
    core: usize,
    tools: usize,
    tail: usize,
    current_reserve: usize,
    output_reserve: usize,
    safety_margin: f64,
) -> usize {
    let evictable =
        context_budget.saturating_sub(core + tools + tail + current_reserve + output_reserve);
    ((evictable as f64) * (1.0 - safety_margin)) as usize
}

pub(crate) fn low_water(archived_budget: usize) -> usize {
    (archived_budget as f64 * 0.60) as usize
}

/// A single archived turn's final-Archived token estimate (input to planning).
pub(crate) struct RenderedTurn {
    pub turn_seq: i64,
    pub est_tokens: usize,
}

pub(crate) struct EvictionPlan {
    pub new_anchor_turn_seq: i64, // turn_seq of the OLDEST kept turn
    pub kept_est_tokens: usize,
    pub evicted_count: usize,
    pub degenerate: bool, // non-evictable >= budget -> zero archived, warn at call site
}

/// `turns` ordered oldest→newest, each with its final-Archived est_tokens.
pub(crate) fn plan_eviction(turns: &[RenderedTurn], archived_budget: usize) -> EvictionPlan {
    let total: usize = turns.iter().map(|t| t.est_tokens).sum();
    if archived_budget == 0 {
        let anchor = turns.last().map(|t| t.turn_seq + 1).unwrap_or(0); // keep nothing
        return EvictionPlan {
            new_anchor_turn_seq: anchor,
            kept_est_tokens: 0,
            evicted_count: turns.len(),
            degenerate: true,
        };
    }
    if total <= archived_budget {
        let anchor = turns.first().map(|t| t.turn_seq).unwrap_or(0);
        return EvictionPlan {
            new_anchor_turn_seq: anchor,
            kept_est_tokens: total,
            evicted_count: 0,
            degenerate: false,
        };
    }
    // Over budget: evict oldest whole turns until kept estimate <= low_water.
    let lw = low_water(archived_budget);
    let mut kept: usize = total;
    let mut evicted = 0;
    for t in turns {
        if kept <= lw {
            break;
        }
        kept -= t.est_tokens;
        evicted += 1;
    }
    // Anchor = the oldest KEPT turn's turn_seq (the first turn not evicted).
    let anchor = turns.get(evicted).map(|t| t.turn_seq).unwrap_or(0);
    EvictionPlan {
        new_anchor_turn_seq: anchor,
        kept_est_tokens: kept,
        evicted_count: evicted,
        degenerate: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archived_budget_and_low_water() {
        let b = archived_budget(
            /*context*/ 32000, /*core*/ 4000, /*tools*/ 3000, /*tail*/ 2000,
            /*current_reserve*/ 4000, /*output_reserve*/ 1536,
            /*safety_margin*/ 0.10,
        );
        // evictable = 32000 - (4000+3000+2000+4000+1536) = 17464; * 0.90 = 15717.6 -> 15717
        assert_eq!(b, 15717);
        assert_eq!(low_water(b), (b as f64 * 0.60) as usize); // 9430
    }

    #[test]
    fn evict_oldest_until_low_water() {
        let budget = 1000usize;
        let turns = vec![
            // oldest first
            RenderedTurn {
                turn_seq: 1,
                est_tokens: 400,
            },
            RenderedTurn {
                turn_seq: 2,
                est_tokens: 400,
            },
            RenderedTurn {
                turn_seq: 3,
                est_tokens: 400,
            },
            RenderedTurn {
                turn_seq: 4,
                est_tokens: 400,
            }, // total 1600 > budget 1000
        ];
        // low_water = 600. Evict oldest whole turns until <= 600.
        // low_water(1000) = 600. kept starts 1600; evict oldest until kept <= 600:
        // 1600 → 1200 (evict t1) → 800 (evict t2) → 400 (evict t3, now <= 600, stop).
        let plan = plan_eviction(&turns, budget);
        assert_eq!(plan.evicted_count, 3);
        assert_eq!(plan.kept_est_tokens, 400);
        assert!(plan.kept_est_tokens <= low_water(budget));
        assert_eq!(plan.new_anchor_turn_seq, 4); // oldest kept turn (turns[3])
        assert!(!plan.degenerate);
    }

    #[test]
    fn degenerate_zero_archived_when_non_evictable_exceeds_budget() {
        // context fully consumed by core+tools+tail+reserves -> archived_budget == 0
        let b = archived_budget(8000, 5000, 2000, 1500, 1000, 1536, 0.10);
        assert_eq!(b, 0);
        let plan = plan_eviction(
            &[RenderedTurn {
                turn_seq: 1,
                est_tokens: 100,
            }],
            b,
        );
        assert_eq!(plan.kept_est_tokens, 0, "zero archived turns carried");
        assert!(
            plan.degenerate,
            "degenerate flag set for warn! at the call site"
        );
    }
}
