pub mod approval;
pub mod background_deliverable;
#[cfg(feature = "browser")]
pub mod browser;
#[cfg(feature = "slack")]
mod channel_history;
mod cli_agent;
pub(crate) mod command_semantics;
#[cfg(feature = "computer_use")]
#[allow(dead_code)]
pub mod computer_use;
mod config_manager;
mod diagnose;
pub mod file_delivery;
mod goal_trace;
mod health_probe;
pub mod image_generation;
mod manage_memories;
mod manage_people;
pub(crate) mod memory;
mod policy_metrics;
pub mod result_spill;
pub mod sanitize;
mod scheduled_goal_runs;
mod search_history;
mod send_file;
mod share_memory;
pub mod spawn;
mod system;
pub mod terminal;
mod tool_trace;
pub mod track_requirements;
pub mod web_fetch;
mod web_search;

pub use approval::ApprovalBroker;
#[cfg(feature = "browser")]
pub use browser::BrowserTool;
#[cfg(feature = "slack")]
pub use channel_history::ReadChannelHistoryTool;
pub use cli_agent::CliAgentTool;
#[cfg(feature = "computer_use")]
pub use computer_use::ComputerUseTool;
pub use config_manager::ConfigManagerTool;
pub use diagnose::DiagnoseTool;
pub use goal_trace::GoalTraceTool;
pub use health_probe::HealthProbeTool;
pub use image_generation::GenerateImageTool;
pub use manage_memories::ManageMemoriesTool;
pub use manage_people::ManagePeopleTool;
pub use memory::RememberFactTool;
pub use policy_metrics::PolicyMetricsTool;
pub use scheduled_goal_runs::ScheduledGoalRunsTool;
pub use search_history::SearchHistoryTool;
pub use send_file::SendFileTool;
pub use share_memory::ShareMemoryTool;
pub use spawn::SpawnAgentTool;
pub use system::SystemInfoTool;
pub use terminal::TerminalTool;
pub use tool_trace::ToolTraceTool;
pub use web_fetch::WebFetchTool;
pub use web_search::WebSearchTool;
mod http_request;
pub use http_request::HttpRequestTool;
mod manage_api;
pub use manage_api::ManageApiTool;
mod manage_http_auth;
pub use manage_http_auth::ManageHttpAuthTool;
mod manage_oauth;
pub use manage_oauth::ManageOAuthTool;
pub mod manage_mcp;
mod manage_skills;
pub mod skill_registry;
mod skill_resources;
mod use_skill;

mod manage_cli_agents;
pub use manage_cli_agents::ManageCliAgentsTool;
pub mod manage_goal_tasks;
pub(crate) use manage_goal_tasks::goal_completion_summary_indicates_not_finished;
pub use manage_goal_tasks::ManageGoalTasksTool;
pub mod report_blocker;
pub use manage_mcp::ManageMcpTool;
pub use manage_skills::ManageSkillsTool;
pub use report_blocker::ReportBlockerTool;
pub use skill_resources::SkillResourcesTool;
pub use use_skill::UseSkillTool;
pub mod command_patterns;
pub mod command_risk;
pub mod daemon_guard;
pub mod verification;
pub use verification::VerificationTracker;

// Deterministic tools
pub(crate) mod fs_utils;
mod read_file;
pub use read_file::ReadFileTool;
pub(crate) use read_file::{render_read_file_output, render_read_file_output_within};
mod write_file;
pub use write_file::WriteFileTool;
mod edit_file;
pub use edit_file::EditFileTool;
mod search_files;
pub use search_files::SearchFilesTool;
mod project_inspect;
pub use project_inspect::ProjectInspectTool;
pub(crate) mod run_command;
pub use run_command::RunCommandTool;
mod git_info;
pub(crate) mod process_control;
pub use git_info::GitInfoTool;
mod git_commit;
pub use git_commit::GitCommitTool;
mod list_checkpoints;
pub use list_checkpoints::ListCheckpointsTool;
mod check_environment;
pub use check_environment::CheckEnvironmentTool;
mod service_status;
pub use service_status::ServiceStatusTool;

#[cfg(test)]
mod schema_lint_tests;
