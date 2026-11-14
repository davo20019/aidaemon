# aidaemon-tools

**Tier: implementation.** No stability promise while 0.x; may change in any minor version. Depend on it through [`aidaemon-sdk`](https://docs.rs/aidaemon-sdk) unless you need it alone.

Ready-made `aidaemon_core::Tool`s: workspace file tools, search, a sandboxed shell, web fetch and a checklist.

```rust,ignore
let workspace = aidaemon_tools::Workspace::new(".")?;
let mut builder = aidaemon_core::Agent::builder("helper"); // ...model, provider
for tool in aidaemon_tools::default_tools(workspace) {
    builder = builder.tool_arc(tool);
}
```

The `shell` tool in `default_tools` runs each command in an OS sandbox confined to the workspace (writes outside it fail, no network unless `Shell::allow_network(true)`): bubblewrap on Linux, seatbelt on macOS. When none is available (Linux without `bwrap`, or with unprivileged user namespaces disabled) it runs commands on the host and logs a one-time warning; its tool description and each call's `backend` detail say `local`. Choose explicitly with `default_tools_with_sandbox(workspace, SandboxKind::Local.backend())` (or any `Sandbox`), or set `AIDAEMON_SHELL_SANDBOX=auto|bubblewrap|seatbelt|local`. An explicitly chosen backend that is unavailable fails each command instead of falling back. Bubblewrap and seatbelt also keep commands from reading the usual credential locations under `$HOME` (`HOME_SECRET_PATHS`: `~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.config/gcloud`, `~/.kube`, `~/.docker/config.json`, `~/.netrc`, `~/.git-credentials`); replace the list with `default_deny_read(..)` on the backend. `Shell::deny_read(paths)` hides more paths. A restriction the backend cannot enforce fails closed: with `Shell::deny_read` or `Shell::allow_network(false)` on the uncontained `local` backend, every command is refused instead of run (`Sandbox::enforces_deny_read`, `Sandbox::enforces_network_off`).

The aidaemon daemon's own general tools also live here, on an `ExecutionBackend` (`LocalBackend` for the host): `web_search` (`web_search::WebSearch`, DuckDuckGo/Brave/SearxNG in parallel), `http_request` (`http_request::HttpRequestTool`, auth profiles and OAuth refresh through `OAuthTokens`), and `system_info`, with the support they share (`command_risk`, `command_effects`, `sanitize`, `fs_utils`, `result_spill`). The browser is its own crate, `aidaemon-tool-browser`. The `testing` feature adds helpers for calling a tool directly from a test.

Part of the [aidaemon SDK](https://github.com/davo20019/aidaemon) (shared version 0.20.0). MIT licensed. Docs: https://docs.rs/aidaemon-tools
