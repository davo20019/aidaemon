# aidaemon-nodes

**Tier: implementation.** No stability promise while 0.x; may change in any minor version. Depend on it through [`aidaemon-sdk`](https://docs.rs/aidaemon-sdk) unless you need it alone.

Nodes are hardware companions (a speaker, a sensor board) enrolled with an agent: each gets a server-issued identity, reports its capabilities, and is authorized action by action. This crate holds their domain (`domain`, the wire models in `protocol`), the SQLite store (`NodeStore`, `store::migrate`), the HTTP Gateway they talk to (`gateway::serve`), pairing and sessions (`NodeService`), signed OTA releases (`ota`), speech for audio replies (`speech`), sensor monitoring with owner alerts (`monitoring`), owner announcements, a simulator, and the tools an agent uses on them: read sensors and health, send audio, manage monitors, restart a Node.

```rust,ignore
let key = aidaemon_nodes::auth::load_or_create_instance_key(secrets.as_ref())?;
let store = Arc::new(aidaemon_nodes::NodeStore::new(pool, key));
let service = Arc::new(aidaemon_nodes::NodeService::new(store, settings.clone()));
let state = aidaemon_nodes::gateway::NodeGatewayState {
    service,
    ingress: Arc::new(MyIngress), // NodeConversationIngress: who answers a Node turn
    hub,
    config: settings,
    rate_limiter: Default::default(),
};
aidaemon_nodes::gateway::serve(state).await?;
```

Settings are `config::NodesConfig` (the `[nodes]` TOML section). The host supplies the conversation behind a Node turn (`NodeConversationIngress`) and the `SecretStore` holding the Gateway instance key.

Part of the [aidaemon SDK](https://github.com/davo20019/aidaemon) (shared version 0.20.0). MIT licensed. Docs: https://docs.rs/aidaemon-nodes
