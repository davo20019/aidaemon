//! `web_search` lives in `aidaemon-tools`. This module holds the daemon's
//! configuration of the tool.

pub use aidaemon_tools::web_search::*;

/// What the daemon's model is told: the crate's description, naming where
/// the owner configures search.
const DAEMON_DESCRIPTION: &str = "Search the web. Queries all configured search backends in parallel and returns merged, deduplicated results (titles, URLs, snippets, publication age when known). Use to find current information, research topics, check facts. One focused search is almost always enough; for single-fact lookups do NOT re-search with rephrased queries — synthesize promptly. Results are snippet previews, NOT full pages: never assemble complete lists, rosters, or enumerations from snippets — fetch a source page with web_fetch and answer from its text. Set 'freshness' for time-sensitive queries (news, prices, releases). If results are consistently empty, the search backend may be blocked — suggest the user set up Brave Search via manage_config (search.backend = 'brave' + search.api_key) or a self-hosted SearxNG instance (search.searxng_url).";

/// The daemon's constructor for the SDK's [`aidaemon_tools::web_search::WebSearch`].
pub struct WebSearchTool;

#[allow(clippy::new_ret_no_self)]
impl WebSearchTool {
    pub fn new(config: &SearchConfig) -> WebSearch {
        WebSearch::new(config).with_description(DAEMON_DESCRIPTION)
    }
}
