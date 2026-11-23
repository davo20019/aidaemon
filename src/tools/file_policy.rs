//! The file tools' policy is `aidaemon_tools::HostFilePolicy`, set up in
//! `startup::tool_config`; the tools and their tests live in `aidaemon-tools`.

pub use aidaemon_tools::path_class::DaemonPaths;

pub(crate) use crate::startup::tool_config::daemon_file_policy;
