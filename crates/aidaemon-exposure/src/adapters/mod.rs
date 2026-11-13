//! The adapters: `manual` (a proxy the owner runs), `cloudflare` (named or
//! quick tunnel), `tailscale` (serve or funnel) and `command` (any program
//! that prints its origin).

pub mod cloudflare;
pub mod command;
pub mod manual;
pub mod tailscale;

use std::path::{Path, PathBuf};

use crate::ExposedService;

/// The pid file of `adapter`'s child for `service`.
pub(crate) fn pidfile(state_dir: &Path, adapter: &str, service: &ExposedService) -> PathBuf {
    state_dir.join(format!("{adapter}-{}.pid", service.key()))
}
