//! Real-Chrome browser smoke tests live in `src/tools/browser/smoke.rs` (library
//! tests) because the browser tool is not part of the public crate API.
//!
//! Run:
//! ```bash
//! cargo test --features browser browser_smoke_real_chrome -- --ignored --nocapture
//! ```

#[test]
fn browser_smoke_entrypoint_documentation() {
    // Keeps this integration-test target present for CI discovery; the ignored
    // real-Chrome suite is `browser_smoke_real_chrome` in the library tests.
}
