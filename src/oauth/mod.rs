//! OAuth lives in `aidaemon_tools::services::oauth`; the daemon keeps its old
//! paths to it. The daemon's gateway is `startup::oauth`, the ChatGPT
//! subscription login `providers::chatgpt_login`.

pub use aidaemon_tools::services::oauth::*;
