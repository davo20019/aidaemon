//! Which ACT/WAIT/ASK/STOP outcomes the current mandate state admits.
//!
//! Each state-dependent outcome rule lives here so the rules can be checked
//! together. Individually correct fail-closed gates composed into dead ends
//! before (stagnation rejected WAIT while the mutation quota rejected ACT,
//! leaving only an ASK the owner could not usefully answer). The liveness
//! invariant tested below is that every state keeps a non-escalating outcome
//! reachable: ACT, or WAIT paired with a strategy adaptation.
//!
//! Malformed decisions (missing receipts, invalid termination fields, and so
//! on) are validated where they are parsed; this module only covers outcomes
//! that a well-formed decision may still be refused because of state.

use crate::traits::MandateMutationQuotaBlockReason;

/// The state inputs that decide outcome admissibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OutcomeState {
    /// Credible measurements stayed flat across the configured no-progress
    /// window, counted once per measurement cadence.
    pub stagnant: bool,
    /// Why a governed mutation cannot be reserved now, if it cannot.
    pub quota_block: Option<MandateMutationQuotaBlockReason>,
    /// The decision revises strategy away from the current tactic (explore,
    /// avoid, or retire), as opposed to reinforcing it.
    pub adapts_strategy: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AdmissibleOutcomes {
    pub act: bool,
    pub wait: bool,
    pub ask: bool,
    pub stop: bool,
}

impl AdmissibleOutcomes {
    /// Human-and-model readable list for refusal messages, so a rejected
    /// decision always names what is still possible instead of a dead end.
    pub(crate) fn describe(&self) -> String {
        let mut parts = Vec::new();
        if self.act {
            parts.push("ACT (a mutation slot is free)");
        }
        if self.wait {
            parts.push("WAIT");
        } else {
            parts.push("WAIT with a strategy revision that explores, avoids, or retires a tactic");
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

pub(crate) fn admissible_outcomes(state: &OutcomeState) -> AdmissibleOutcomes {
    let quota_temporarily_blocked = matches!(
        state.quota_block,
        Some(
            MandateMutationQuotaBlockReason::Rolling24hExhausted
                | MandateMutationQuotaBlockReason::Cooldown
        )
    );
    AdmissibleOutcomes {
        act: state.quota_block.is_none(),
        // The no-progress rule exists to stop passive waiting. Waiting for the
        // next owner-granted mutation slot is not passive, and neither is a
        // WAIT that changes tactics; both remain admissible when stagnant.
        wait: !state.stagnant || quota_temporarily_blocked || state.adapts_strategy,
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

    fn every_state() -> impl Iterator<Item = OutcomeState> {
        [false, true].into_iter().flat_map(|stagnant| {
            QUOTA_STATES.into_iter().flat_map(move |quota_block| {
                [false, true]
                    .into_iter()
                    .map(move |adapts_strategy| OutcomeState {
                        stagnant,
                        quota_block,
                        adapts_strategy,
                    })
            })
        })
    }

    #[test]
    fn every_state_keeps_a_non_escalating_outcome_reachable() {
        for state in every_state() {
            let adapting = admissible_outcomes(&OutcomeState {
                adapts_strategy: true,
                ..state
            });
            let admissible = admissible_outcomes(&state);
            assert!(
                admissible.act || adapting.wait,
                "{state:?}: neither ACT nor an adapting WAIT is admissible"
            );
            assert!(admissible.ask && admissible.stop, "{state:?}");
        }
    }

    #[test]
    fn a_refused_act_always_leaves_wait_admissible() {
        // The ACT refusal tells the model to choose WAIT; that advice must
        // never point at an outcome the store would reject.
        for state in every_state().filter(|state| {
            matches!(
                state.quota_block,
                Some(
                    MandateMutationQuotaBlockReason::Rolling24hExhausted
                        | MandateMutationQuotaBlockReason::Cooldown
                )
            )
        }) {
            assert!(admissible_outcomes(&state).wait, "{state:?}");
        }
    }

    #[test]
    fn stagnation_still_refuses_a_passive_wait_when_action_is_possible() {
        let state = OutcomeState {
            stagnant: true,
            quota_block: None,
            adapts_strategy: false,
        };
        let admissible = admissible_outcomes(&state);
        assert!(!admissible.wait);
        assert!(admissible.act);
        assert!(admissible
            .describe()
            .contains("WAIT with a strategy revision"));
    }
}
