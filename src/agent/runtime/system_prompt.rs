//! The system prompt lives in `aidaemon-runtime`
//! ([`aidaemon_runtime::prompt::system`]).

#[cfg(test)]
use super::*;
#[allow(unused_imports)]
pub(in crate::agent) use aidaemon_runtime::prompt::system::*;

pub(in crate::agent) use aidaemon_runtime::prompt::rules::infer_assistant_name_from_prompt;

/// Render the "## Available Specialists" block surfaced in the agent's system
/// prompt. Mirrors the per-kind list also exposed via the `spawn_agent` tool
/// schema, so the LLM has two consistent surfaces to discover which specialist
/// profiles exist and what each one is for.
///
/// Driven by the live `SpecialistRegistry` — user overrides at
/// `~/.aidaemon/specialists/<kind>.md` flow into this block on next start.
///
/// `task_lead` is intentionally omitted (role-typed, assigned by the agent,
/// not parent-LLM-selectable). Returns an empty string only if the registry
/// is empty, which should never happen by construction; the caller can drop
/// the section entirely in that case.
///
/// As of Pillar A Task 4 the production splice is performed by
/// `core_prompt::render_core_prompt` (via `render_specialists_block`) over the
/// pre-extracted `llm_visible_kinds()` pairs; this registry-driven variant is
/// retained as the test oracle for that block's byte format.
#[cfg(test)]
pub(crate) fn build_available_specialists_block(
    registry: &crate::agent::specialists::SpecialistRegistry,
) -> String {
    let entries = registry.llm_visible_kinds();
    if entries.is_empty() {
        return String::new();
    }

    let mut s = String::from(
        "## Available Specialists\n\n\
         When you delegate work with `spawn_agent`, pick the specialist that best matches the task. \
         Sub-agents run in an isolated context window with the same tools you have, so keep the `mission` \
         and `task` brief minimal — reference files by path rather than pasting contents, and skip prior \
         tool output or conversation history the sub-agent does not need:\n\n",
    );
    for (name, description) in &entries {
        s.push_str("- `");
        s.push_str(name);
        s.push_str("`: ");
        s.push_str(description);
        if !description.ends_with('.') {
            s.push('.');
        }
        s.push('\n');
    }
    s.push_str(
        "\nOmit the `specialist` argument to let the agent infer the right kind from the mission/task text.",
    );
    s
}

#[cfg(test)]
impl Agent {
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn build_system_prompt_for_message(
        &self,
        emitter: &crate::events::EventEmitter,
        task_id: &str,
        session_id: &str,
        user_text: &str,
        user_role: UserRole,
        channel_ctx: &ChannelContext,
        tools_count: usize,
        resume_checkpoint: Option<&ResumeCheckpoint>,
        preserve_conversation_context: bool,
        project_instruction_scope: Option<&str>,
    ) -> anyhow::Result<(
        String,
        String,
        String,
        Vec<String>,
        Option<aidaemon_runtime::turn::project_instructions::ProjectInstructionTracker>,
    )> {
        build_system_prompt_for_message(
            self,
            emitter,
            task_id,
            session_id,
            user_text,
            user_role,
            channel_ctx,
            tools_count,
            resume_checkpoint,
            preserve_conversation_context,
            project_instruction_scope,
        )
        .await
    }

    pub(crate) fn core_behavioral_rules() -> String {
        aidaemon_runtime::prompt::rules::core_behavioral_rules()
    }

    fn build_context_tail(
        project_instructions_block: Option<&str>,
        skills_content: &str,
        sender_name: Option<&str>,
        session_context_str: &str,
        date_time_str: &str,
        resume_section: Option<&str>,
    ) -> String {
        build_context_tail(
            project_instructions_block,
            skills_content,
            sender_name,
            session_context_str,
            date_time_str,
            resume_section,
        )
    }
}

#[cfg(test)]
pub(super) use aidaemon_runtime::prompt::rules::{render_environment_section, task_lead_rules};

#[cfg(test)]
mod tests {
    use super::{
        build_available_specialists_block, infer_assistant_name_from_prompt, task_lead_rules,
    };
    use aidaemon_runtime::prompt::rules::format_goal_context;

    /// A bounded mandate's lead carries no control-loop rules and no
    /// receipt-copying instructions; autopilot with an objective control
    /// keeps the measured loop.
    #[test]
    fn bounded_task_lead_prompt_omits_control_loop_rules() {
        // The single task-lead rule string before the split.
        const PREVIOUS_RULES_CHARS: usize = 5_800;
        let bounded = task_lead_rules("Bounded mode is active.", false);
        assert!(!bounded.contains("record_measurement"), "{bounded}");
        assert!(!bounded.contains("proposes a complete objective_control"));
        assert!(!bounded.contains("retain their exact successful receipt identifiers"));
        assert!(
            bounded.len() * 10 <= PREVIOUS_RULES_CHARS * 7,
            "bounded rules are {} chars",
            bounded.len()
        );
        let autopilot = task_lead_rules("Autopilot is owner-confirmed.", true);
        assert!(autopilot.contains("record_measurement"), "{autopilot}");
    }

    /// The rules name manage_mandates arguments; each must be one the tool's
    /// schema accepts, or a model that follows the prompt is refused by
    /// `additionalProperties: false`.
    #[tokio::test]
    async fn task_lead_rules_name_only_real_manage_mandates_arguments() {
        use crate::traits::Tool;
        let harness = crate::testing::setup_test_agent(crate::testing::MockProvider::new())
            .await
            .unwrap();
        let tool = crate::tools::ManageMandatesTool::new(harness.state.clone());
        let schema = tool.schema();
        let properties = schema["parameters"]["properties"]
            .as_object()
            .expect("manage_mandates properties");
        let rules = task_lead_rules("Autopilot is owner-confirmed.", true);
        for argument in [
            "evidence_receipt_ids",
            "activity_level",
            "value_criterion",
            "expected_benefit",
            "risk",
            "invalidation_criteria",
            "intention",
            "termination_kind",
            "reconsider_minutes",
            "measurement_value_micros",
            "measurement_confidence_bps",
            "learning_note",
        ] {
            assert!(rules.contains(argument), "rules no longer name {argument}");
            assert!(
                properties.contains_key(argument),
                "rules name {argument}, which manage_mandates does not accept"
            );
        }
        for retired in [
            "record value_micros",
            "with value_micros",
            " confidence_bps,",
        ] {
            assert!(!rules.contains(retired), "rules still say {retired:?}");
        }
    }

    #[test]
    fn configured_multi_word_agent_name_is_inferred_from_identity_line() {
        let prompt = "## Identity\nYou are Project Nova, a personal AI assistant.";
        assert_eq!(
            infer_assistant_name_from_prompt(prompt).as_deref(),
            Some("Project Nova")
        );
    }

    /// Byte-identity guard for Pillar A Task 4: the production CORE base prompt
    /// Renderer-level guard: with EMPTY `channel_rules`/`skills_catalog` the
    /// `render_core_prompt` output equals the legacy `base_prompt` construction
    /// (persona + the `## Available Specialists` block spliced before `## Tools`)
    /// byte-for-byte. NOTE (Pillar A Task 6): the production call site now passes
    /// the REAL channel_rules + skills_catalog (so the production core is larger
    /// than legacy by design); this test pins the renderer's empty-input contract
    /// only, not the production bytes.
    #[test]
    fn render_core_prompt_matches_legacy_base_prompt_construction() {
        use crate::agent::core_prompt::{assemble_core_inputs, render_core_prompt};
        use crate::types::{ChannelContext, UserRole};

        let registry = crate::agent::specialists::SpecialistRegistry::load(None);

        // A persona stand-in that contains a `## Tools` anchor, mirroring the
        // real `self.system_prompt`.
        let persona = "You are aidaemon.\n\n## Behavior\nBe helpful.\n\n## Tools\nUse them.";

        // --- legacy construction (non-public path) ---
        let specialists_block = build_available_specialists_block(&registry);
        let legacy = if specialists_block.is_empty() {
            persona.to_string()
        } else if let Some(idx) = persona.find("## Tools") {
            let (head, tail) = persona.split_at(idx);
            format!("{head}{specialists_block}\n\n{tail}")
        } else {
            format!("{persona}\n\n{specialists_block}")
        };

        // --- new construction via the assembler + pure renderer ---
        let core_inputs = assemble_core_inputs(
            UserRole::Owner,
            &ChannelContext::private("test"),
            persona.to_string(),
            Vec::new(),
            Vec::new(),
            registry
                .llm_visible_kinds()
                .into_iter()
                .map(|(n, d)| (n.to_string(), d))
                .collect(),
            String::new(),
        );
        let rendered = render_core_prompt(&core_inputs);

        assert_eq!(
            rendered, legacy,
            "render_core_prompt must reproduce the legacy base_prompt byte-for-byte"
        );
    }

    #[test]
    fn available_specialists_block_lists_each_non_task_lead_kind() {
        let registry = crate::agent::specialists::SpecialistRegistry::load(None);
        let block = build_available_specialists_block(&registry);

        // Section header is present so downstream prompt-shapers can find it.
        assert!(
            block.contains("## Available Specialists"),
            "missing header: {}",
            block
        );

        // Every parent-LLM-selectable kind appears as its own bullet.
        for kind in [
            "code",
            "browser_verifier",
            "artifact_writer",
            "research",
            "review",
            "comms_draft",
            "executor",
            "generic",
        ] {
            let bullet = format!("- `{}`:", kind);
            assert!(
                block.contains(&bullet),
                "missing bullet for {}: {}",
                kind,
                block
            );
        }

        // task_lead is role-typed and must NOT appear in the LLM-facing list.
        assert!(!block.contains("- `task_lead`:"));
        assert!(!block.contains("`task_lead`"));

        // Sanity: the actual frontmatter description for `code` flowed into
        // the block (proves it's data-driven, not a static string).
        let code_def = registry.get(crate::traits::SpecialistKind::Code);
        assert!(
            block.contains(&code_def.description),
            "code description not surfaced: {}",
            block
        );

        // Closing line tells the model omission is allowed.
        assert!(block.contains("Omit the `specialist` argument"));
    }

    #[test]
    fn format_goal_context_includes_recent_messages_and_project_hints() {
        let ctx = serde_json::json!({
            "relevant_facts": [],
            "relevant_procedures": [],
            "recent_messages": [
                {"role": "user", "content": "Please modernize test-project with Tailwind."},
                {"role": "assistant", "content": "Which sections should I update?"}
            ],
            "project_hints": ["test-project"],
            "task_results": []
        });

        let formatted = format_goal_context(&ctx.to_string());
        assert!(formatted.contains("### Project Hints"));
        assert!(formatted.contains("test-project"));
        assert!(formatted.contains("### Recent Parent Conversation"));
        assert!(formatted.contains("[user] Please modernize test-project with Tailwind."));
    }

    #[tokio::test]
    async fn new_task_prompt_omits_prior_summary_and_session_activity() {
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::types::{ChannelContext, UserRole};

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("setup test harness");
        let emitter = crate::events::EventEmitter::new(
            harness.agent.event_store().clone(),
            "new-task-session".to_string(),
        );
        let (_, tail, continuation_tail, _, _) = harness
            .agent
            .build_system_prompt_for_message(
                &emitter,
                "new-task",
                "new-task-session",
                "Observe the current command result.",
                UserRole::Owner,
                &ChannelContext::private("synthetic-owner"),
                0,
                None,
                false,
                None,
            )
            .await
            .expect("build new-task prompt");

        assert!(!tail.contains("STALE_OUTPUT_FORMAT"));
        assert!(!tail.contains("package_name=stale"));
        assert!(!tail.contains("Compacted Conversation State"));
        assert!(!tail.contains("Recent Session Context"));
        // A continuation turn's summary is placed by the SDK summary note
        // (`conversation_compaction::summary_note`), never in either tail.
        assert!(!continuation_tail.contains("Compacted Conversation State"));
    }

    #[tokio::test]
    async fn mandate_context_uses_only_built_in_policy_prompt_inputs() {
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::{
            Goal, Mandate, MandateAuthority, MandateObjectiveControl, MandateStore,
            ObjectiveMetricDirection, TaskAttempt,
        };
        use crate::types::{ChannelContext, UserRole};

        let mut harness = setup_test_agent(MockProvider::new())
            .await
            .expect("setup test harness");
        harness.agent.system_prompt =
            "PRIVATE CUSTOM PERSONA SENTINEL\nPRIVATE MCP SCHEMA SENTINEL".to_string();

        let skill_dir = tempfile::tempdir().expect("skill tempdir");
        std::fs::write(
            skill_dir.path().join("private-skill.md"),
            "---\nname: private-skill\ndescription: PRIVATE SKILL DESCRIPTION SENTINEL\ntriggers: [private-skill]\n---\nPRIVATE SKILL BODY SENTINEL\n",
        )
        .expect("write custom skill");
        harness.agent.skills_dir = skill_dir.path().to_path_buf();
        harness.agent.skill_cache = crate::skills::SkillCache::new(skill_dir.path().to_path_buf());

        let specialist_dir = tempfile::tempdir().expect("specialist tempdir");
        std::fs::write(
            specialist_dir.path().join("code.md"),
            "---\nkind: code\ndescription: PRIVATE SPECIALIST DESCRIPTION SENTINEL\n---\nPRIVATE SPECIALIST BODY SENTINEL\n",
        )
        .expect("write custom specialist override");
        harness.agent.specialists = std::sync::Arc::new(
            crate::agent::specialists::SpecialistRegistry::load(Some(specialist_dir.path())),
        );

        let project_dir = tempfile::tempdir().expect("project tempdir");
        std::fs::write(
            project_dir.path().join("AGENTS.md"),
            "PRIVATE PROJECT INSTRUCTION SENTINEL",
        )
        .expect("write project instruction");

        let attempt = TaskAttempt {
            id: "attempt-privacy".to_string(),
            task_id: "task-privacy".to_string(),
            goal_run_id: "run-privacy".to_string(),
            worker_profile_id: Some("profile-task-lead".to_string()),
            worker_instance_id: "worker-privacy".to_string(),
            lease_token: "lease-secret".to_string(),
            status: "running".to_string(),
            lease_expires_at: (chrono::Utc::now() + chrono::Duration::minutes(3)).to_rfc3339(),
            last_heartbeat_at: chrono::Utc::now().to_rfc3339(),
            workspace_id: None,
            started_at: chrono::Utc::now().to_rfc3339(),
            completed_at: None,
        };
        let authority = MandateAuthority {
            allow_observations: true,
            allowed_tools: vec!["http_request".to_string()],
            allowed_mutation_effects: vec![
                "remote_mutation".to_string(),
                "external_delivery".to_string(),
            ],
            allowed_target_prefixes: vec!["https://api.x.com/2/".to_string()],
            operation_scopes: Vec::new(),
            max_mutating_actions_per_cycle: 1,
            max_mutating_actions_per_rolling_24h: 8,
            min_seconds_between_mutations: 1_800,
            owner_alerts: false,
        };
        let mut goal = Goal::new_continuous(
            "Mandate prompt test controller",
            "owner-private-session",
            Some(10_000),
            Some(50_000),
        );
        goal.id = "goal-privacy".to_string();
        goal.context = Some(
            serde_json::json!({
                "mandate_id": "mandate-privacy",
                "owner_guidance": [{
                    "guidance": "OWNER GUIDANCE SENTINEL",
                    "recorded_at": "2026-08-01T00:00:00Z"
                }],
                "unrelated_private_context": "PRIVATE GOAL CONTEXT SENTINEL",
                "history_like_private_context": {
                    "rationale": "PRIVATE HISTORY RATIONALE SENTINEL",
                    "arguments": "PRIVATE HISTORY ARGUMENTS SENTINEL"
                }
            })
            .to_string(),
        );
        let mut mandate = Mandate::new(
            &goal.id,
            None,
            "OWNER POLICY OBJECTIVE SENTINEL",
            "owner-private-session",
            authority.clone(),
            900,
            14_400,
            3_600,
        );
        mandate.id = "mandate-privacy".to_string();
        mandate.version = 1;
        mandate.autonomy_mode = crate::traits::MandateAutonomyMode::Autopilot;
        mandate.objective_control = Some(MandateObjectiveControl {
            schema_version: MandateObjectiveControl::SCHEMA_VERSION,
            metric_name: "synthetic public engagement".to_string(),
            unit: "count".to_string(),
            baseline_micros: 1_000_000,
            target_micros: 2_000_000,
            direction: ObjectiveMetricDirection::AtLeast,
            measurement_source: "metric_source:synthetic-public-analytics".to_string(),
            measurement_cadence_secs: 3_600,
            baseline_observed_at: "2026-08-01T00:00:00Z".to_string(),
        });
        mandate.constraints = vec!["OWNER POLICY CONSTRAINT SENTINEL".to_string()];
        mandate.success_criteria = vec!["OWNER POLICY SUCCESS SENTINEL".to_string()];
        mandate.stop_conditions = vec!["OWNER POLICY STOP SENTINEL".to_string()];
        harness
            .state
            .create_mandate_controller(&goal, &mandate)
            .await
            .expect("persist mandate prompt policy");
        harness.agent.set_test_mandate_execution(
            "mandate-privacy",
            1,
            authority.clone(),
            "goal-privacy",
            "task-privacy",
            &attempt.id,
            &attempt,
        );
        let emitter = crate::events::EventEmitter::new(
            harness.agent.event_store().clone(),
            "mandate-session".to_string(),
        );
        let (core, tail, continuation_tail, active_skills, project_tracker) = harness
            .agent
            .build_system_prompt_for_message(
                &emitter,
                "prompt-task",
                "mandate-session",
                "$private-skill Review current public engagement",
                UserRole::Owner,
                &ChannelContext::internal(),
                1,
                None,
                false,
                project_dir.path().to_str(),
            )
            .await
            .expect("build mandate prompt");
        let prompt = format!("{core}\n{tail}");
        assert_eq!(tail, continuation_tail);
        for private_sentinel in [
            "PRIVATE CUSTOM PERSONA SENTINEL",
            "PRIVATE MCP SCHEMA SENTINEL",
            "PRIVATE SKILL DESCRIPTION SENTINEL",
            "PRIVATE SKILL BODY SENTINEL",
            "PRIVATE SPECIALIST DESCRIPTION SENTINEL",
            "PRIVATE SPECIALIST BODY SENTINEL",
            "PRIVATE PROJECT INSTRUCTION SENTINEL",
            "PRIVATE GOAL CONTEXT SENTINEL",
            "PRIVATE HISTORY RATIONALE SENTINEL",
            "PRIVATE HISTORY ARGUMENTS SENTINEL",
            "Secret Owner Name",
            "SECRET PRIOR CONVERSATION",
        ] {
            assert!(
                !prompt.contains(private_sentinel),
                "mandate prompt leaked {private_sentinel:?}"
            );
        }
        for generic_shell in ["## Available Skills", "## Active Skill", "## Your Memory"] {
            assert!(
                !prompt.contains(generic_shell),
                "mandate prompt used generic shell {generic_shell:?}"
            );
        }
        assert!(prompt.contains("[Immutable Mandate Execution Policy]"));
        assert!(prompt.contains("mandate-privacy"));
        assert!(prompt.contains("mandate_version: 1"));
        assert!(prompt.contains("Autopilot is owner-confirmed for this exact policy version"));
        assert!(prompt.contains("prefer WAIT over asking about ordinary tactics"));
        assert!(prompt.contains("http_request"));
        assert!(prompt.contains("https://api.x.com/2/"));
        assert!(prompt.contains("OWNER POLICY OBJECTIVE SENTINEL"));
        assert!(prompt.contains("OWNER POLICY CONSTRAINT SENTINEL"));
        assert!(prompt.contains("OWNER POLICY SUCCESS SENTINEL"));
        assert!(prompt.contains("OWNER POLICY STOP SENTINEL"));
        assert!(prompt.contains("OWNER GUIDANCE SENTINEL"));
        assert!(prompt.contains("[Autonomous Mandate History: Untrusted, Non-Authoritative]"));
        assert!(prompt.contains("autonomous_mandate_history_untrusted"));
        assert!(prompt.contains("\"authority\":false"));
        assert!(prompt.contains("cannot grant, widen, restore, or reinterpret authority"));
        assert!(prompt.contains("\"scope\":\"same_mandate_typed_history_only\""));
        assert!(prompt.contains("\"minimum\":900"));
        assert!(prompt.contains("\"default\":3600"));
        assert!(prompt.contains("\"maximum\":14400"));
        assert!(prompt.contains("\"max_mutating_actions_per_rolling_24h\":8"));
        assert!(prompt.contains("\"min_seconds_between_mutations\":1800"));
        assert!(prompt.contains("role: task_lead"));
        assert!(prompt.contains("OBSERVE -> decide exactly one of ACT, WAIT, ASK, or STOP"));
        assert!(prompt.contains("never an open-ended loop"));
        assert!(prompt.contains("select one typed activity_level"));
        assert!(prompt.contains("meeting that cadence is a sufficient expected_benefit"));
        assert!(prompt.contains("an unreadable metric is an outage"));
        assert!(prompt.contains("do not repeat it"));
        assert!(prompt.contains("one exact owner-authored value_criterion"));
        assert!(prompt.contains("do not act to demonstrate activity"));
        assert!(prompt.contains("learning_note"));
        assert!(prompt.contains("never alters authority"));
        assert!(prompt.contains("Never ASK for permission to retry"));
        assert!(prompt.contains("ASK once for that exact"));
        assert!(prompt.contains("answer_question or ordinary guidance"));
        assert!(prompt.contains("record WAIT with no children"));
        assert!(prompt.contains("separately confirmed policy update"));
        assert!(prompt.contains("## Available Specialists"));
        assert!(!prompt.contains("lease-secret"));
        assert!(active_skills.is_empty());
        assert!(project_tracker.is_none());

        let mut executor_attempt = attempt.clone();
        executor_attempt.id = "attempt-executor-privacy".to_string();
        executor_attempt.task_id = "task-executor-privacy".to_string();
        use crate::traits::store_prelude::*;
        harness
            .state
            .create_task(&crate::traits::Task {
                id: "task-executor-privacy".to_string(),
                goal_id: "goal-privacy".to_string(),
                description: "Publish exactly: EXACT_POST_TEXT_SENTINEL".to_string(),
                status: "claimed".to_string(),
                priority: "high".to_string(),
                task_order: 1,
                parallel_group: None,
                depends_on: None,
                agent_id: None,
                context: None,
                result: None,
                error: None,
                blocker: None,
                idempotent: false,
                retry_count: 0,
                max_retries: 0,
                created_at: chrono::Utc::now().to_rfc3339(),
                started_at: None,
                completed_at: None,
            })
            .await
            .expect("create executor work task");
        executor_attempt.lease_token = "executor-lease-secret".to_string();
        harness.agent.set_test_mandate_execution(
            "mandate-privacy",
            1,
            authority,
            "goal-privacy",
            "task-privacy",
            &attempt.id,
            &executor_attempt,
        );
        let (
            executor_core,
            executor_tail,
            executor_continuation_tail,
            executor_skills,
            executor_tracker,
        ) = harness
            .agent
            .build_system_prompt_for_message(
                &emitter,
                "executor-prompt-task",
                "mandate-executor-session",
                "$private-skill Execute the committed action",
                UserRole::Owner,
                &ChannelContext::internal(),
                1,
                None,
                false,
                project_dir.path().to_str(),
            )
            .await
            .expect("build isolated mandate executor prompt");
        let executor_prompt = format!("{executor_core}\n{executor_tail}");
        assert_eq!(executor_tail, executor_continuation_tail);
        assert!(executor_prompt.contains("role: executor"));
        assert!(
            executor_prompt.contains("EXACT_POST_TEXT_SENTINEL"),
            "the executor must see the exact work it was claimed for"
        );
        assert!(!prompt.contains("EXACT_POST_TEXT_SENTINEL"));
        assert!(executor_prompt.contains("durable current-run successful receipt"));
        assert!(executor_prompt.contains("cannot observe, read, or preflight"));
        assert!(!executor_prompt.contains("role: task_lead"));
        // What binds the call stays; what the lead deliberated with does not.
        assert!(executor_prompt.contains("OWNER POLICY CONSTRAINT SENTINEL"));
        assert!(executor_prompt.contains("OWNER POLICY STOP SENTINEL"));
        assert!(executor_prompt.contains("https://api.x.com/2/"));
        assert!(!executor_prompt.contains("autonomous_mandate_history_untrusted"));
        assert!(!executor_prompt.contains("OWNER POLICY SUCCESS SENTINEL"));
        assert!(!executor_prompt.contains("OWNER GUIDANCE SENTINEL"));
        assert!(!executor_prompt.contains("Available Specialists"));
        assert!(prompt.contains("Available Specialists"));
        assert!(!executor_prompt.contains("executor-lease-secret"));
        assert!(executor_skills.is_empty());
        assert!(executor_tracker.is_none());
    }

    // ---- Pillar A Task 5: context tail builder ----

    use crate::agent::prefix_fingerprint::TASK_CONTEXT_TAIL_MARKER;
    use crate::agent::Agent;

    /// The tail starts with the marker and carries the volatile sections —
    /// here we assert the date/time block and the marker.
    #[test]
    fn environment_section_names_backend_os_and_home() {
        let section = super::render_environment_section(
            crate::execution::BackendKind::Local,
            "/Users/synthetic-user",
        );
        assert!(section.starts_with("[Environment]"));
        assert!(section.contains("Home directory: /Users/synthetic-user."));
        assert!(section.contains(std::env::consts::OS));
        let remote = super::render_environment_section(
            crate::execution::BackendKind::Ssh,
            "/home/synthetic-user",
        );
        assert!(!remote.contains(std::env::consts::OS));
        assert!(remote.contains("/home/synthetic-user"));
    }

    #[test]
    fn context_tail_carries_all_volatile_sections_and_marker() {
        let tail =
            Agent::build_context_tail(None, "", None, "", "Monday, June 1, 2026 12:00 UTC", None);
        assert!(tail.starts_with(TASK_CONTEXT_TAIL_MARKER));
        let needle = "[Current Date & Time]";
        assert!(tail.contains(needle), "missing {needle}");
    }

    /// Spec §Tail: the resume checkpoint MOVES from the core to the tail.
    /// Assert it lands in the tail and is ABSENT from `render_core_prompt`.
    #[test]
    fn resume_checkpoint_renders_into_tail_not_core() {
        use crate::agent::core_prompt::{render_core_prompt, test_core_inputs};

        let checkpoint_section =
            "## Resume Checkpoint\nThe user explicitly asked to continue prior in-progress work.";
        let tail = Agent::build_context_tail(
            None,
            "",
            None,
            "",
            "Monday, June 1, 2026 12:00 UTC",
            Some(checkpoint_section),
        );
        assert!(
            tail.contains(checkpoint_section),
            "resume checkpoint must render into the tail"
        );

        let core = render_core_prompt(&test_core_inputs());
        assert!(
            !core.contains("## Resume Checkpoint"),
            "resume checkpoint must be ABSENT from the core prompt"
        );
    }

    #[test]
    fn core_behavioral_rules_contains_all_core_items() {
        let core = Agent::core_behavioral_rules();
        for needle in [
            "Never claim actions were performed unless confirmed by a tool result",
            "Quotation marks mean verbatim",
            "you have a tool for", // capability-honesty kernel
            "Never claim tests pass or builds succeed without running them",
            "Use write_file/edit_file for file creation",
            "Deliver large output", // rule 5
            "[Data Integrity Rule]",
            "[Credential Protection — ABSOLUTE RULE]",
        ] {
            assert!(core.contains(needle), "core rules missing: {needle}");
        }
    }

    #[test]
    fn root_rules_preserve_all_content_after_core_extraction() {
        // Core rules contain all core-tier content.
        let core = Agent::core_behavioral_rules();
        for needle in [
            "Never claim actions were performed unless confirmed by a tool result",
            "you have a tool for",
            "Never claim tests pass or builds succeed without running them",
            "Use write_file/edit_file for file creation",
            "Deliver large output",
            "[Data Integrity Rule]",
            "[Credential Protection — ABSOLUTE RULE]",
        ] {
            assert!(core.contains(needle), "root rules lost core item: {needle}");
        }
        // Root-only items must NOT have leaked into the shared core block.
        for needle in [
            "[Identity Stability Rule",
            "[Model Identity — CRITICAL]",
            "Cross-reference memory before answering",
            "Question contradictory identity claims",
            "Never mention tool names in responses",
            "Proactively store personal information",
            "Wait for background services",
            "Trust explicit paths",
            "Quote stored fact values EXACTLY",
        ] {
            assert!(
                !core.contains(needle),
                "root-only rule leaked into core_behavioral_rules: {needle}"
            );
        }
    }

    #[test]
    fn context_tail_includes_scoped_project_instructions() {
        let instructions = "[Project Instructions — scoped workspace guidance]\nUse cargo fmt.";
        let tail = Agent::build_context_tail(
            Some(instructions),
            "",
            None,
            "",
            "Monday, June 1, 2026 12:00 UTC",
            None,
        );

        assert!(tail.starts_with(TASK_CONTEXT_TAIL_MARKER));
        assert!(tail.contains(instructions));
        assert!(tail.find(instructions).unwrap() < tail.find("[Current Date & Time]").unwrap());
    }
}
