pub mod approval;
#[cfg(feature = "slack")]
#[cfg(feature = "slack")]
mod channel_history;
pub(crate) use aidaemon_runtime::tools::cli_agent;
#[cfg(feature = "computer_use")]
pub use aidaemon_runtime::tools::computer_use;
pub(crate) mod config_manager;
#[allow(unused_imports)]
pub(crate) use aidaemon_tools::consequential;
mod diagnose;
/// SDK tools as daemon tools (no SDK tool is wrapped yet; 8.4 moves them).
#[allow(dead_code)]
mod file_policy;
pub use file_policy::DaemonPaths;
pub(crate) use file_policy::daemon_file_policy;
#[cfg(test)]
pub(crate) use file_policy::{guarded_call, guarded_call_protecting};
mod goal_trace;
pub mod image_generation;
pub(crate) mod manage_automations;
pub(crate) mod manage_mandates;
mod manage_memories;
pub(crate) mod objective_status;
mod policy_metrics;
pub use aidaemon_tools::result_spill;
pub use aidaemon_tools::sanitize;
mod scheduled_goal_runs;
mod search_history;
mod send_file;
mod set_reminder;
pub mod spawn;
pub use aidaemon_runtime::tools::terminal;
mod tool_trace;
pub mod track_requirements;
pub mod web_fetch;
mod web_search;

#[cfg(feature = "browser")]
pub use aidaemon_tool_browser::BrowserTool;
pub use approval::ApprovalBroker;
#[cfg(feature = "slack")]
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
#[cfg(test)]
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

pub use aidaemon_runtime::tools::manage_cli_agents::ManageCliAgentsTool;
pub mod manage_goal_tasks;
#[cfg(test)]
pub use manage_goal_tasks::ManageGoalTasksTool;
pub mod report_blocker;
pub use aidaemon_client_gateway::client_view::ClientViewTool;
pub use aidaemon_skills::tools::{ManageSkillsTool, SkillResourcesTool, UseSkillTool};
pub use aidaemon_tools::command_risk;
pub use manage_mcp::ManageMcpTool;

// Deterministic tools
pub(crate) use aidaemon_tools::fs_utils;
mod read_file;
pub use crate::startup::tool_config::ReadFileTool;
mod write_file;
pub use crate::startup::tool_config::WriteFileTool;
mod edit_file;
pub use crate::startup::tool_config::EditFileTool;
mod search_files;
pub use crate::startup::tool_config::SearchFilesTool;
pub(crate) use aidaemon_runtime::tools::run_command;
pub use aidaemon_tools::check_environment::CheckEnvironmentTool;
pub use aidaemon_tools::checkpoints::ListCheckpointsTool;
pub use aidaemon_tools::git_commit::GitCommitTool;
pub use aidaemon_tools::git_info::GitInfoTool;
pub use aidaemon_tools::project_inspect::ProjectInspectTool;
pub use aidaemon_tools::service_status::ServiceStatusTool;
pub use run_command::RunCommandTool;

#[cfg(test)]
mod catalog_lint_tests;
#[cfg(test)]
mod execution_backend_injection_tests;
#[cfg(test)]
mod schema_golden_tests;
#[cfg(test)]
mod schema_lint_tests;
#[cfg(test)]
mod work_schema_golden_tests;
