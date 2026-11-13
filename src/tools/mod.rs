pub mod approval;
pub use aidaemon_tools::background_deliverable;
#[cfg(feature = "slack")]
mod channel_history;
pub(crate) mod cli_agent;
#[cfg(feature = "computer_use")]
#[allow(dead_code)]
pub mod computer_use;
pub(crate) mod config_manager;
#[allow(unused_imports)]
pub(crate) use aidaemon_tools::consequential;
mod diagnose;
pub use aidaemon_tools::file_delivery;
/// SDK tools as daemon tools (no SDK tool is wrapped yet; 8.4 moves them).
#[allow(dead_code)]
mod file_policy;
mod goal_ids;
pub(crate) use file_policy::daemon_file_policy;
pub use file_policy::DaemonPaths;
#[cfg(test)]
pub(crate) use file_policy::{guarded_call, guarded_call_protecting};
mod goal_trace;
pub mod image_generation;
pub(crate) mod manage_automations;
pub(crate) mod manage_mandates;
mod manage_memories;
pub(crate) mod objective_status;
pub(crate) use aidaemon_tools::owner_answers;
mod policy_metrics;
pub use aidaemon_tools::result_spill;
pub use aidaemon_tools::sanitize;
mod scheduled_goal_runs;
mod search_history;
mod send_file;
mod set_reminder;
pub mod spawn;
pub mod terminal;
mod tool_trace;
pub mod track_requirements;
pub mod web_fetch;
mod web_search;

#[cfg(feature = "browser")]
pub use aidaemon_tool_browser::BrowserTool;
pub use approval::ApprovalBroker;
pub(crate) use approval::{OwnerWaitClock, ProgressClock};
#[cfg(feature = "slack")]
pub use channel_history::ReadChannelHistoryTool;
pub use cli_agent::CliAgentTool;
#[cfg(feature = "computer_use")]
pub use computer_use::ComputerUseTool;
pub use config_manager::ConfigManagerTool;
mod escalate;
pub use aidaemon_memory::full::tools::ShareMemoryTool;
pub use aidaemon_memory::full::tools::{ManagePeopleTool, RememberFactTool};
pub use aidaemon_tools::http_request;
pub use aidaemon_tools::system::SystemInfoTool;
pub use aidaemon_work::health::HealthProbeTool;
pub use diagnose::DiagnoseTool;
pub use escalate::EscalateToParentTool;
pub use goal_trace::GoalTraceTool;
pub use http_request::HttpRequestTool;
pub use image_generation::GenerateImageTool;
pub use manage_automations::ManageAutomationsTool;
pub use manage_mandates::ManageMandatesTool;
pub use manage_memories::ManageMemoriesTool;
pub use policy_metrics::PolicyMetricsTool;
pub use scheduled_goal_runs::ScheduledGoalRunsTool;
pub use search_history::SearchHistoryTool;
pub use send_file::SendFileTool;
pub use set_reminder::SetReminderTool;
pub use spawn::SpawnAgentTool;
pub use terminal::TerminalTool;
pub use tool_trace::ToolTraceTool;
pub use web_fetch::WebFetchTool;
pub use web_search::WebSearchTool;
mod manage_api;
pub use manage_api::ManageApiTool;
mod manage_http_auth;
pub use manage_http_auth::ManageHttpAuthTool;
mod manage_oauth;
pub use manage_oauth::ManageOAuthTool;
pub mod manage_mcp;

mod manage_cli_agents;
pub use manage_cli_agents::ManageCliAgentsTool;
pub mod manage_goal_tasks;
pub use manage_goal_tasks::ManageGoalTasksTool;
pub mod client_view;
pub mod report_blocker;
pub use aidaemon_skills::tools::{ManageSkillsTool, SkillResourcesTool, UseSkillTool};
pub(crate) use aidaemon_tools::command_effects;
pub use client_view::ClientViewTool;
pub use manage_mcp::ManageMcpTool;
pub use report_blocker::ReportBlockerTool;
pub mod command_patterns;
pub use aidaemon_tools::command_risk;
pub use aidaemon_tools::daemon_guard;
mod semantic_command_risk;

// Deterministic tools
pub(crate) use aidaemon_tools::fs_utils;
mod read_file;
pub use read_file::ReadFileTool;
mod write_file;
pub use write_file::WriteFileTool;
mod edit_file;
pub use edit_file::EditFileTool;
mod search_files;
pub use aidaemon_tools::project_inspect::ProjectInspectTool;
pub use search_files::SearchFilesTool;
pub(crate) mod run_command;
pub use aidaemon_tools::checkpoints::ListCheckpointsTool;
pub use aidaemon_tools::git_commit::GitCommitTool;
pub use aidaemon_tools::git_info::GitInfoTool;
pub(crate) use aidaemon_tools::process_control;
pub use run_command::RunCommandTool;
mod check_environment;
pub use aidaemon_tools::service_status::ServiceStatusTool;
pub use check_environment::CheckEnvironmentTool;

#[cfg(test)]
mod execution_backend_injection_tests;
#[cfg(test)]
mod schema_golden_tests;
#[cfg(test)]
mod schema_lint_tests;
#[cfg(test)]
mod work_schema_golden_tests;
