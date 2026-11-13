/// Scope checks, evidence matching and acks of a dispatched call live in
/// `aidaemon-runtime`.
#[cfg(test)]
mod run_helpers {
    pub(in crate::agent) use aidaemon_runtime::turn::tool_dispatch::*;
}
#[cfg(test)]
mod send_file_scope_tests;

#[cfg(test)]
pub(in crate::agent) use run_helpers::access_manifest_scope_check;
#[cfg(test)]
pub(in crate::agent) use run_helpers::ScopeCheck;
