//! # aidaemon-tool-browser
//!
//! The `browser` tool for aidaemon agents: drive a Chrome instance (launched
//! headless, or an attached remote-debugging Chrome) to navigate, read pages,
//! click, fill forms, take screenshots and download files. Consequential
//! actions (submitting, purchasing, running scripts on a page) ask the owner
//! through the agent kernel's approval flow; per-session isolation is honest
//! about whether cookies are shared ([`SessionIsolation`]).
//!
//! The tool is behind the `chromium` feature; without it the crate has only
//! [`BrowserSettings`], so a host can parse and validate its configuration
//! without building Chrome support.

pub mod settings;

pub use settings::{BrowserSettings, SessionIsolation};

#[cfg(feature = "chromium")]
mod tool;

#[cfg(feature = "chromium")]
pub use tool::{policy, BrowserTool};
