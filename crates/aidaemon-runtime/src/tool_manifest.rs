//! The one declared list of every tool the daemon registers: its name, the
//! context it belongs to, where it is built and when it is enabled.
//!
//! Tools are constructed where their dependencies live (startup's
//! registries, the skills setup, the checklist in `core.rs`, the host's
//! per-run tools, a child's own tools), but none of those sites decides on
//! its own that a tool exists: each registers through [`push`] or checks
//! its list with [`check_declared`], so a tool missing here fails in debug
//! builds and tests. The schema lint's catalog is built from this list
//! (`startup::tools::lint_catalog::build_full` in the daemon), and its test
//! checks every declared tool reaches the catalog and every catalog tool is
//! declared.

use std::sync::Arc;

use aidaemon_tools::OutcomeTool;

/// Which runs a tool is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToolContext {
    /// Built once at startup into the root tool list; any run may be
    /// offered it, as its role, roster and authority allow.
    Global,
    /// Added to a goal's task-lead run, bound to that goal.
    TaskLead,
    /// Added to a mandate review's task-lead run.
    MandateReview,
    /// Added to the executor run of a claimed task attempt.
    TaskAttempt,
    /// Added to a child agent's own tools.
    Child,
    /// Added to a plain child of a coordinated tree.
    CoordinatedChild,
}

/// Where a tool is constructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltAt {
    /// `startup::tools` `BASE_TOOL_REGISTRY` (and `list_checkpoints` beside it).
    StartupBase,
    /// `startup::tools` `OPTIONAL_TOOL_REGISTRY`.
    StartupOptional,
    /// `startup::tools` `RUNTIME_TOOL_MANIFEST`.
    StartupRuntime,
    /// `startup::skills::register_skills_tools`.
    StartupSkills,
    /// `core.rs`, after every other startup tool (it binds the subjects
    /// they advertise).
    StartupChecklist,
    /// The host, for one run (`TurnHost::goal_tasks_tool`,
    /// `mandate_review_tool`, `report_blocker_tool`).
    HostPerRun,
    /// `Agent::create_child_agent`.
    ChildAgent,
}

/// When a tool is registered, beyond `[tools].disabled`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Enabled {
    /// Whenever its site runs.
    Always,
    /// When this config holds (startup) or this run fact holds (per run).
    When(&'static str),
}

/// Whether the schema lint covers a tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lint {
    /// In the lint catalog, with a golden and a budget line.
    Catalog,
    /// Left out of the catalog, for this reason.
    Skipped(&'static str),
}

/// One declared tool.
#[derive(Debug, Clone, Copy)]
pub struct ToolEntry {
    pub name: &'static str,
    pub contexts: &'static [ToolContext],
    pub built_at: BuiltAt,
    pub enabled: Enabled,
    /// The daemon's cargo feature the tool is compiled under, if any.
    pub feature: Option<&'static str>,
    pub lint: Lint,
}

const GLOBAL: &[ToolContext] = &[ToolContext::Global];

const fn startup(name: &'static str, built_at: BuiltAt, enabled: Enabled) -> ToolEntry {
    ToolEntry {
        name,
        contexts: GLOBAL,
        built_at,
        enabled,
        feature: None,
        lint: Lint::Catalog,
    }
}

const fn base(name: &'static str) -> ToolEntry {
    startup(name, BuiltAt::StartupBase, Enabled::Always)
}

const fn optional(name: &'static str, when: &'static str) -> ToolEntry {
    startup(name, BuiltAt::StartupOptional, Enabled::When(when))
}

const fn per_run(
    name: &'static str,
    context: &'static [ToolContext],
    built_at: BuiltAt,
    when: &'static str,
) -> ToolEntry {
    ToolEntry {
        name,
        contexts: context,
        built_at,
        enabled: Enabled::When(when),
        feature: None,
        lint: Lint::Catalog,
    }
}

const CLI_AGENT_SKIP: Lint =
    Lint::Skipped("discovers the CLI agents installed on the machine it runs on");

/// Every tool the daemon registers.
pub const TOOL_MANIFEST: &[ToolEntry] = &[
    base("system_info"),
    base("terminal"),
    base("remember_fact"),
    base("manage_memories"),
    base("set_reminder"),
    base("schedule_task"),
    base("manage_mandates"),
    base("manage_automations"),
    base("search_history"),
    base("manage_config"),
    base("web_fetch"),
    base("web_search"),
    base("read_file"),
    base("write_file"),
    base("edit_file"),
    base("search_files"),
    base("check_environment"),
    startup(
        "list_checkpoints",
        BuiltAt::StartupBase,
        Enabled::When("a checkpoint manager is installed ([checkpoints])"),
    ),
    optional("self_diagnose", "diagnostics.enabled"),
    optional("goal_trace", "diagnostics.enabled"),
    optional("policy_metrics", "diagnostics.enabled"),
    optional("read_node_health", "nodes.gateway.enabled"),
    optional("read_node_sensors", "nodes.gateway.enabled"),
    optional(
        "send_node_audio",
        "nodes.gateway, nodes.announcements and nodes.speech enabled",
    ),
    optional("node_restart", "nodes.gateway.enabled"),
    optional(
        "manage_node_monitors",
        "nodes.gateway and nodes.monitoring enabled",
    ),
    ToolEntry {
        feature: Some("browser"),
        ..optional("browser", "browser.enabled")
    },
    ToolEntry {
        feature: Some("computer_use"),
        ..optional("computer_use", "computer_use.enabled")
    },
    optional("generate_image", "files.enabled"),
    optional("send_file", "files.enabled"),
    ToolEntry {
        lint: CLI_AGENT_SKIP,
        ..optional("cli_agent", "cli_agents.enabled")
    },
    ToolEntry {
        lint: CLI_AGENT_SKIP,
        ..optional("manage_cli_agents", "cli_agents.enabled")
    },
    optional("health_probe", "a health probe store is configured"),
    ToolEntry {
        lint: Lint::Skipped("reads the client gateway's live views"),
        ..optional("client_view", "client_gateway.enabled")
    },
    ToolEntry {
        feature: Some("slack"),
        ..optional("read_channel_history", "always (Slack build)")
    },
    startup("manage_mcp", BuiltAt::StartupRuntime, Enabled::Always),
    startup("manage_people", BuiltAt::StartupRuntime, Enabled::Always),
    startup("http_request", BuiltAt::StartupRuntime, Enabled::Always),
    startup("manage_auth", BuiltAt::StartupRuntime, Enabled::Always),
    startup("manage_api", BuiltAt::StartupRuntime, Enabled::Always),
    ToolEntry {
        contexts: &[ToolContext::Global, ToolContext::Child],
        ..startup(
            "spawn_agent",
            BuiltAt::StartupRuntime,
            Enabled::When("subagents.enabled; for a child, when it may spawn its own"),
        )
    },
    startup("use_skill", BuiltAt::StartupSkills, Enabled::Always),
    startup("skill_resources", BuiltAt::StartupSkills, Enabled::Always),
    startup("manage_skills", BuiltAt::StartupSkills, Enabled::Always),
    startup(
        "track_requirements",
        BuiltAt::StartupChecklist,
        Enabled::Always,
    ),
    per_run(
        "manage_goal_tasks",
        &[ToolContext::TaskLead],
        BuiltAt::HostPerRun,
        "the run leads a goal's tasks",
    ),
    per_run(
        "mandate_review",
        &[ToolContext::MandateReview],
        BuiltAt::HostPerRun,
        "the task lead reviews a mandate",
    ),
    per_run(
        "report_blocker",
        &[ToolContext::TaskAttempt],
        BuiltAt::HostPerRun,
        "the executor holds a claimed task attempt",
    ),
    per_run(
        "escalate_to_parent",
        &[ToolContext::CoordinatedChild],
        BuiltAt::ChildAgent,
        "the child's tree has a coordinator",
    ),
];

/// The entry for `name`, if declared.
pub fn entry(name: &str) -> Option<&'static ToolEntry> {
    TOOL_MANIFEST.iter().find(|entry| entry.name == name)
}

/// Whether `name` is declared for `context`.
pub fn is_declared(name: &str, context: ToolContext) -> bool {
    entry(name).is_some_and(|entry| entry.contexts.contains(&context))
}

/// The names in `names` not declared for `context`.
pub fn undeclared<'a>(
    names: impl IntoIterator<Item = &'a str>,
    context: ToolContext,
) -> Vec<&'a str> {
    names
        .into_iter()
        .filter(|name| !is_declared(name, context))
        .collect()
}

/// Check a site's registered tools against the manifest: an undeclared one
/// fails a debug build and is logged in a release build (it still runs:
/// the manifest declares, it does not gate).
pub fn check_declared(tools: &[Arc<dyn OutcomeTool>], context: ToolContext) {
    let missing = undeclared(tools.iter().map(|tool| tool.name()), context);
    if !missing.is_empty() {
        tracing::warn!(
            ?missing,
            ?context,
            "Tools registered without a manifest entry"
        );
        debug_assert!(
            false,
            "tools registered for {context:?} without a tool_manifest entry: {missing:?}"
        );
    }
}

/// Add `tool` to a run's tools, declared for `context`.
pub fn push(
    tools: &mut Vec<Arc<dyn OutcomeTool>>,
    context: ToolContext,
    tool: Arc<dyn OutcomeTool>,
) {
    check_declared(std::slice::from_ref(&tool), context);
    tools.push(tool);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tool_is_declared_once_with_a_context() {
        let mut seen = std::collections::HashSet::new();
        for entry in TOOL_MANIFEST {
            assert!(seen.insert(entry.name), "{} declared twice", entry.name);
            assert!(!entry.contexts.is_empty(), "{} has no context", entry.name);
        }
    }

    #[test]
    fn a_run_scoped_tool_is_declared_for_its_run_only() {
        assert!(is_declared("report_blocker", ToolContext::TaskAttempt));
        assert!(!is_declared("report_blocker", ToolContext::Global));
        assert!(is_declared("spawn_agent", ToolContext::Child));
        assert!(is_declared("spawn_agent", ToolContext::Global));
        assert_eq!(
            undeclared(["terminal", "made_up_tool"], ToolContext::Global),
            ["made_up_tool"]
        );
    }
}
