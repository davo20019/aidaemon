//! `web_fetch` and the network policy it enforces live in `aidaemon-tools`.
//! This module holds the daemon's configuration of the tool.

pub(crate) use aidaemon_tools::web::read_body_capped;
pub use aidaemon_tools::web::{build_browser_client, validate_url_for_ssrf};
use aidaemon_tools::ToolText;

/// The daemon's constructors for the SDK's [`aidaemon_tools::WebFetch`].
pub struct WebFetchTool;

#[allow(clippy::new_ret_no_self)]
impl WebFetchTool {
    pub fn new() -> aidaemon_tools::WebFetch {
        aidaemon_tools::WebFetch::new()
                .browser_identity()
                .follow_public_redirects()
                .with_text(ToolText {
                    short: "Fetch a readable web page and extract its content; not for REST/JSON API endpoints".to_string(),
                    schema: "Fetch a readable web page and extract its content. Strips ads/navigation. Do NOT use for REST/JSON API endpoints or machine-readable responses; use http_request for APIs. For login-required sites, use browser instead.".to_string(),
                })
    }
}
