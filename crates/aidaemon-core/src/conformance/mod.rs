//! Conformance kits (feature `testing`): reusable checks that third-party
//! [`Tool`](crate::Tool) and [`Provider`](crate::Provider) implementations
//! behave the way the kernel expects.
//!
//! Each kit runs every check and returns a [`ConformanceReport`] instead of
//! panicking on the first problem, so an author sees all failures at once.
//! Call [`ConformanceReport::assert_all_passed`] in a test to turn the report
//! into a single panic that lists every failure.
//!
//! ```
//! use aidaemon_core::conformance::check_tool;
//! use aidaemon_core::testing::spec;
//! use aidaemon_core::{FnTool, ToolAnnotations, ToolOutput};
//!
//! let echo = FnTool::new(
//!     spec("echo", "Echo the text back", &["text"]),
//!     ToolAnnotations::READ_ONLY,
//!     |_args| async { Ok(ToolOutput::text("ok")) },
//! );
//! check_tool(&echo).assert_all_passed();
//! ```
//!
//! Check names are stable identifiers (`tool.name_format`,
//! `provider.stream_well_formed/plain`, ...). Each one is documented on the
//! matching constant in [`tool_checks`] and [`provider_checks`]. Sibling
//! crates build their own kits on the same report type
//! (`aidaemon_channels::conformance`, `aidaemon_memory::conformance`).

mod provider;
mod report;
mod tool;

pub use provider::{
    check_provider, probe_requests, provider_checks, PROVIDER_CALLS_PER_REQUEST,
    PROVIDER_CALL_TIMEOUT,
};
pub use report::ConformanceReport;
pub use tool::{check_tool, check_tool_with, sample_arguments, tool_checks, ToolKitOptions};
