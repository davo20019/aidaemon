//! Which ACT/WAIT/ASK/STOP outcomes the current mandate state admits.
//!
//! Each state-dependent outcome rule lives here so the rules can be checked
//! together. Individually correct fail-closed gates composed into dead ends
//! before. The liveness invariant tested below is that every state keeps a
//! non-escalating outcome reachable: ACT, or WAIT.
//!
//! Malformed decisions (missing receipts, invalid termination fields, and so
//! on) are validated where they are parsed; this module only covers outcomes
//! that a well-formed decision may still be refused because of state.

use crate::traits::MandateMutationQuotaBlockReason;

/// The state inputs that decide outcome admissibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutcomeState {
    /// Why a governed mutation cannot be reserved now, if it cannot.
    pub quota_block: Option<MandateMutationQuotaBlockReason>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissibleOutcomes {
    pub act: bool,
    pub wait: bool,
    pub ask: bool,
    pub stop: bool,
}

impl AdmissibleOutcomes {
    /// Human-and-model readable list for refusal messages, so a rejected
    /// decision always names what is still possible instead of a dead end.
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        if self.act {
            parts.push("ACT (a mutation slot is free)");
        }
        if self.wait {
            parts.push("WAIT");
        }
        if self.ask {
            parts.push("ASK");
        }
        if self.stop {
            parts.push("STOP");
        }
        parts.join(", ")
    }
}

pub fn admissible_outcomes(state: &OutcomeState) -> AdmissibleOutcomes {
    AdmissibleOutcomes {
        act: state.quota_block.is_none(),
        wait: true,
        ask: true,
        stop: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QUOTA_STATES: [Option<MandateMutationQuotaBlockReason>; 4] = [
        None,
        Some(MandateMutationQuotaBlockReason::MutationDisabled),
        Some(MandateMutationQuotaBlockReason::Rolling24hExhausted),
        Some(MandateMutationQuotaBlockReason::Cooldown),
    ];

    #[test]
    fn every_state_keeps_a_non_escalating_outcome_reachable() {
        for quota_block in QUOTA_STATES {
            let admissible = admissible_outcomes(&OutcomeState { quota_block });
            assert!(admissible.act || admissible.wait, "{quota_block:?}");
            assert!(admissible.ask && admissible.stop, "{quota_block:?}");
            assert_eq!(admissible.act, quota_block.is_none());
        }
    }
}
