//! The registry of side calls: model calls the daemon makes outside the agent
//! loop. Each call has one spec saying which model it uses, how long it may
//! take, what happens when it fails, and the purpose string its telemetry is
//! recorded under.

use std::time::Duration;

/// A model call made outside the agent loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SideCall {
    /// Checks a proposed egress call against the original request after the
    /// run read untrusted content.
    UntrustedVerifier,
    /// Judges the risk of a terminal command before it runs.
    CommandRisk,
    /// Decides how a new message relates to earlier tasks.
    RelationshipRouter,
    /// Summarizes old conversation turns during compaction.
    Summarization,
    /// Turns a finished background command's output into a plain sentence.
    BackgroundInterpretation,
    /// Extracts memory facts from each exchange as the conversation goes.
    ProgressiveExtraction,
    /// Extracts memory facts after a turn through the SDK memory plugin.
    FactExtraction,
    /// Learns reusable lessons from a finished task.
    TaskLearning,
    /// Merges and deduplicates stored facts.
    FactConsolidation,
    /// Analyzes a finished session into an episode.
    SessionAnalysis,
    /// Improves learned procedures from consolidated events.
    ProcedureEnhancement,
    /// Writes a skill from a procedure that earned promotion.
    SkillPromotion,
    /// Answers a sub-agent's escalation, as its request's top-level agent.
    ParentCoordination,
    /// Answers a decision's closed questions on a chat model, when the
    /// decision backend is `provider` (see `llm_runtime::decisions`).
    DecisionAsk,
}

/// What a side call's answer is used for; this sets how it may fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SideCallClass {
    /// Its answer decides whether an action may happen. A judge tries the
    /// strongest model (the primary) first, then the failover models, and
    /// moves on only after an infrastructure failure.
    SafetyJudge,
    /// Its answer shapes the input the agent loop sees.
    InputShaper,
    /// Its answer only changes how a result is shown to the user.
    Presentation,
    /// Background work on memory; nobody waits for it.
    MemoryJob,
    /// Its answer becomes something the agent later runs (procedures, skills).
    ExecutableProducer,
}

/// Which model a side call runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelChoice {
    /// The configured primary model. For a safety judge, the first model
    /// it tries.
    Primary,
    /// The configured cheap model (the router's fast tier).
    Cheap,
    /// The caller passes the model in.
    CallerSupplied,
}

/// What happens when a side call errors, times out or returns nothing usable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnFailure {
    /// Refuse the action the call was judging.
    Refuse,
    /// Continue with a conservative default answer.
    SafeDefault,
    /// Show the raw, uninterpreted result.
    RawFallback,
    /// Skip the work; it is retried or dropped later.
    Skip,
}

/// How one side call runs.
#[derive(Debug, Clone, Copy)]
pub struct SideCallSpec {
    /// The purpose string its telemetry is recorded under.
    pub purpose: &'static str,
    /// What its answer is used for.
    pub class: SideCallClass,
    /// Which model it runs on.
    pub model: ModelChoice,
    /// How long it may take before it counts as failed.
    pub deadline: Duration,
    /// What happens when it fails. Each caller carries the policy out
    /// itself; the registry tests hold the classes to it.
    #[cfg(test)]
    pub on_failure: OnFailure,
}

impl SideCall {
    /// Every side call, in registry order.
    #[cfg(test)]
    pub const ALL: [SideCall; 14] = [
        SideCall::UntrustedVerifier,
        SideCall::CommandRisk,
        SideCall::RelationshipRouter,
        SideCall::Summarization,
        SideCall::BackgroundInterpretation,
        SideCall::ProgressiveExtraction,
        SideCall::FactExtraction,
        SideCall::TaskLearning,
        SideCall::FactConsolidation,
        SideCall::SessionAnalysis,
        SideCall::ProcedureEnhancement,
        SideCall::SkillPromotion,
        SideCall::ParentCoordination,
        SideCall::DecisionAsk,
    ];

    /// This call's spec.
    pub const fn spec(self) -> SideCallSpec {
        use ModelChoice::*;
        use OnFailure::*;
        use SideCallClass::*;
        let (purpose, class, model, secs, _on_failure) = match self {
            SideCall::UntrustedVerifier => {
                ("untrusted_call_verifier", SafetyJudge, Primary, 20, Refuse)
            }
            SideCall::CommandRisk => ("terminal_command_risk", SafetyJudge, Primary, 15, Refuse),
            SideCall::RelationshipRouter => {
                ("task_relationship", InputShaper, Cheap, 45, SafeDefault)
            }
            SideCall::Summarization => ("summarization", InputShaper, Cheap, 60, SafeDefault),
            SideCall::BackgroundInterpretation => (
                "background_interpretation",
                Presentation,
                Primary,
                30,
                RawFallback,
            ),
            SideCall::ProgressiveExtraction => {
                ("progressive_extraction", MemoryJob, Cheap, 120, Skip)
            }
            SideCall::FactExtraction => ("sdk_memory_extraction", MemoryJob, Cheap, 120, Skip),
            SideCall::TaskLearning => ("task_learning", MemoryJob, CallerSupplied, 120, Skip),
            SideCall::FactConsolidation => ("memory_consolidation", MemoryJob, Cheap, 120, Skip),
            SideCall::SessionAnalysis => ("episode_creation", MemoryJob, Cheap, 120, Skip),
            SideCall::ProcedureEnhancement => {
                ("consolidation", ExecutableProducer, Primary, 180, Skip)
            }
            SideCall::SkillPromotion => ("skill_promotion", ExecutableProducer, Cheap, 180, Skip),
            SideCall::ParentCoordination => {
                ("parent_coordination", SafetyJudge, Primary, 90, Refuse)
            }
            // The decision's own deadline bounds it too; whoever asked
            // falls back to its default on failure.
            SideCall::DecisionAsk => ("decision_ask", InputShaper, CallerSupplied, 10, SafeDefault),
        };
        SideCallSpec {
            purpose,
            class,
            model,
            deadline: Duration::from_secs(secs),
            #[cfg(test)]
            on_failure: _on_failure,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_side_call_has_a_unique_purpose() {
        let purposes: std::collections::HashSet<_> =
            SideCall::ALL.iter().map(|c| c.spec().purpose).collect();
        assert_eq!(purposes.len(), SideCall::ALL.len());
    }

    #[test]
    fn safety_judges_refuse_and_start_from_the_primary_model() {
        for call in SideCall::ALL {
            let spec = call.spec();
            assert_eq!(
                spec.class == SideCallClass::SafetyJudge,
                spec.on_failure == OnFailure::Refuse
            );
            if spec.class == SideCallClass::SafetyJudge {
                assert_eq!(spec.model, ModelChoice::Primary);
            }
        }
    }

    #[test]
    fn deadlines_match_the_design_table() {
        use std::time::Duration;
        assert_eq!(
            SideCall::UntrustedVerifier.spec().deadline,
            Duration::from_secs(20)
        );
        assert_eq!(
            SideCall::CommandRisk.spec().deadline,
            Duration::from_secs(15)
        );
        assert_eq!(
            SideCall::RelationshipRouter.spec().deadline,
            Duration::from_secs(45)
        );
        assert_eq!(
            SideCall::BackgroundInterpretation.spec().deadline,
            Duration::from_secs(30)
        );
        assert_eq!(
            SideCall::TaskLearning.spec().deadline,
            Duration::from_secs(120)
        );
        assert_eq!(
            SideCall::SkillPromotion.spec().deadline,
            Duration::from_secs(180)
        );
    }
}
