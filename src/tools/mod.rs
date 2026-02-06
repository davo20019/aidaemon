#[cfg(feature = "browser")]
pub mod browser;
mod cli_agent;
mod config_manager;
mod health_probe;
mod memory;
mod plan_manager;
mod scheduler;
mod send_file;
pub mod spawn;
mod system;
pub mod terminal;
pub mod web_fetch;
mod web_search;

#[cfg(feature = "browser")]
pub use browser::BrowserTool;
pub use cli_agent::CliAgentTool;
pub use config_manager::ConfigManagerTool;
pub use health_probe::HealthProbeTool;
pub use memory::RememberFactTool;
pub use plan_manager::PlanManagerTool;
pub use scheduler::SchedulerTool;
pub use send_file::SendFileTool;
pub use spawn::SpawnAgentTool;
pub use system::SystemInfoTool;
pub use terminal::TerminalTool;
pub use web_fetch::WebFetchTool;
pub use web_search::WebSearchTool;
mod manage_skills;
pub mod skill_registry;
mod use_skill;

pub use manage_skills::ManageSkillsTool;
pub use use_skill::UseSkillTool;
pub mod command_patterns;
pub mod command_risk;
