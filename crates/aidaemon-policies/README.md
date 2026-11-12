# aidaemon-policies

**Tier: implementation.** No stability promise while 0.x; may change in any minor version. Depend on it through [`aidaemon-sdk`](https://docs.rs/aidaemon-sdk) unless you need it alone.

Hook-based policies for `aidaemon-core` agents: approvals (`ApprovalPolicy`, `ApprovalMode`), scope and egress guards, a same-tool streak guard, redaction, untrusted-output handling, context-window management, memory injection and tracing (`otel` feature).

```rust,ignore
use aidaemon_policies::{ApprovalMode, ApprovalPolicy};
let agent = builder.hook(ApprovalPolicy::new(ApprovalMode::DestructiveOnly)).build()?;
```

Part of the [aidaemon SDK](https://github.com/davo20019/aidaemon) (shared version 0.1.0). MIT licensed. Docs: https://docs.rs/aidaemon-policies
