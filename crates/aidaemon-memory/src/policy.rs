//! The automatic-memory policy of one task, compiled once from typed facts.
//! Prompt retrieval, message-span projection and post-task learning all read
//! this one value; none of them reinterprets the user's wording.

/// Who wrote the turn's text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnAuthor {
    /// A person wrote it.
    Person,
    /// No person wrote it (background continuation, runtime resume, mandate
    /// cycle, control command).
    Synthetic,
}

/// What automatic memory may do on one task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPipelinePolicy {
    Allowed,
    /// The request the task continues forbade user memory.
    SuppressedByCurrentContract,
    /// No person wrote this turn's text. Recall stays on so the agent still
    /// knows who it works for; persistence stays off so it cannot learn from
    /// its own synthetic text.
    RecallOnlyNonUserTurn,
}

impl MemoryPipelinePolicy {
    /// Compile the policy: a request that forbade user memory suppresses it;
    /// otherwise a person's turn keeps full memory and a turn no person wrote
    /// may only recall.
    pub const fn compile(author: TurnAuthor, user_memory_forbidden: bool) -> Self {
        if user_memory_forbidden {
            Self::SuppressedByCurrentContract
        } else if matches!(author, TurnAuthor::Person) {
            Self::Allowed
        } else {
            Self::RecallOnlyNonUserTurn
        }
    }

    /// Whether automatic memory may be written or learned from this task.
    pub const fn allows_memory(self) -> bool {
        matches!(self, Self::Allowed)
    }

    /// Whether stored memory may be recalled into this task's prompt.
    pub const fn allows_recall(self) -> bool {
        matches!(self, Self::Allowed | Self::RecallOnlyNonUserTurn)
    }

    pub const fn reason_code(self) -> &'static str {
        match self {
            Self::Allowed => "allowed_by_compiled_contract",
            Self::SuppressedByCurrentContract => "inherited_contract_denies_user_memory",
            Self::RecallOnlyNonUserTurn => "non_user_turn_recall_only",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_forbidding_request_wins_and_synthetic_turns_only_recall() {
        let person = MemoryPipelinePolicy::compile(TurnAuthor::Person, false);
        assert!(person.allows_memory() && person.allows_recall());
        let synthetic = MemoryPipelinePolicy::compile(TurnAuthor::Synthetic, false);
        assert!(synthetic.allows_recall() && !synthetic.allows_memory());
        for author in [TurnAuthor::Person, TurnAuthor::Synthetic] {
            let denied = MemoryPipelinePolicy::compile(author, true);
            assert!(!denied.allows_recall() && !denied.allows_memory());
        }
    }
}
