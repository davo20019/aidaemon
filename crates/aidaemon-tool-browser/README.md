# aidaemon-tool-browser

**Tier: implementation.** No stability promise while 0.x; may change in any minor version. Depend on it through [`aidaemon-sdk`](https://docs.rs/aidaemon-sdk) unless you need it alone.

The `browser` tool: Chrome over the DevTools protocol (`chromiumoxide`), launched headless or attached to a remote-debugging Chrome. It navigates, reads pages, clicks, fills forms, takes screenshots and downloads files, and asks the owner (through the agent kernel's approval flow) before consequential actions. Enable it with the `chromium` feature; without it the crate only has `BrowserSettings` and `SessionIsolation`.

```rust,ignore
let tool = aidaemon_tool_browser::BrowserTool::new(settings, media_tx, inbox_dir, backend)?;
```

Part of the [aidaemon SDK](https://github.com/davo20019/aidaemon) (shared version 0.2.0). MIT licensed. Docs: https://docs.rs/aidaemon-tool-browser
