#[cfg(feature = "browser")]
pub mod browser;
#[cfg(feature = "slack")]
mod channel_history;
mod cli_agent;
mod config_manager;
mod health_probe;
mod manage_memories;
mod manage_people;
pub(crate) mod memory;
mod plan_manager;
pub mod sanitize;
mod scheduler;
mod send_file;
mod share_memory;
pub mod spawn;
mod system;
pub mod terminal;
pub mod web_fetch;
mod web_search;

#[cfg(feature = "browser")]
pub use browser::BrowserTool;
#[cfg(feature = "slack")]
pub use channel_history::ReadChannelHistoryTool;
pub use cli_agent::CliAgentTool;
pub use config_manager::ConfigManagerTool;
pub use health_probe::HealthProbeTool;
pub use manage_memories::ManageMemoriesTool;
pub use manage_people::ManagePeopleTool;
pub use memory::RememberFactTool;
pub use plan_manager::PlanManagerTool;
pub use scheduler::SchedulerTool;
pub use send_file::SendFileTool;
pub use share_memory::ShareMemoryTool;
pub use spawn::SpawnAgentTool;
pub use system::SystemInfoTool;
pub use terminal::TerminalTool;
pub use web_fetch::WebFetchTool;
pub use web_search::WebSearchTool;
mod http_request;
pub use http_request::HttpRequestTool;
mod manage_oauth;
pub use manage_oauth::ManageOAuthTool;
pub mod manage_mcp;
mod manage_skills;
pub mod skill_registry;
mod skill_resources;
mod use_skill;

pub use manage_mcp::ManageMcpTool;
pub use manage_skills::ManageSkillsTool;
pub use skill_resources::SkillResourcesTool;
pub use use_skill::UseSkillTool;
pub mod command_patterns;
pub mod command_risk;
pub mod verification;
pub use verification::VerificationTracker;
