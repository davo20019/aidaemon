//! `web_fetch` and the network policy it enforces live in `aidaemon-tools`.
//! This module holds the daemon's configuration of the tool.

pub use aidaemon_tools::web::{
    build_browser_client, classify_blocked_host, local_site_key, validate_url_for_ssrf,
    BlockedHostClass,
};
pub(crate) use aidaemon_tools::web::{read_body_capped, MAX_FETCH_BODY_BYTES};
use aidaemon_tools::ToolText;

use super::kernel_tool::KernelTool;

pub type WebFetchTool = KernelTool<aidaemon_tools::WebFetch>;

impl WebFetchTool {
    pub fn new() -> Self {
        Self::wrap(
            aidaemon_tools::WebFetch::new()
                .browser_identity()
                .follow_public_redirects()
                .with_text(ToolText {
                    short: "Fetch a readable web page and extract its content; not for REST/JSON API endpoints".to_string(),
                    schema: "Fetch a readable web page and extract its content. Strips ads/navigation. Do NOT use for REST/JSON API endpoints or machine-readable responses; use http_request for APIs. For login-required sites, use browser instead.".to_string(),
                }),
        )
    }
}
