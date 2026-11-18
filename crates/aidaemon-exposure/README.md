# aidaemon-exposure

**Tier: implementation.** No stability promise while 0.x; may change in any minor version.

Publish a loopback gateway (the Client Gateway on `/client/v1`, the Node Gateway on `/node/v1`) through a tunnel the daemon supervises, and feed the live HTTPS origin to the gateway through an `aidaemon_core::EndpointSource`, so the owner never copies a URL by hand or runs a tunnel agent under launchd or systemd.

An adapter moves bytes; it never authenticates. Device keys, signed challenges, scopes and risk ceilings stay in the gateway protocols.

- `ExposureAdapter` / `Exposure`: publish one loopback listener; the handle's `watch()` carries `ExposureHealth` (`Starting`, `Up`, `Degraded`, `Down`) as the origin or health changes.
- `ExposureSupervisor`: one exposure per service, capped exponential backoff (1 s to 60 s, reset after 5 minutes up), every reported origin checked by `validate_https_origin`, ephemeral origins refused for the Node Gateway, only registered loopback listeners (`ServiceListeners`) exposed. While an exposure is down the gateway serves its static config origins.
- Tunnels live and die with the daemon: process adapters run their program as a foreground child, stopped with the daemon.

```rust,ignore
let (source, publisher) = aidaemon_core::EndpointSource::live(static_origins);
// hand `source` to the gateway, then:
let mut listeners = aidaemon_exposure::ServiceListeners::new();
listeners.register(ExposedService::ClientGateway, "127.0.0.1:8788".parse()?)?;
let supervisor = ExposureSupervisor::spawn(
    vec![ExposurePlan { service: ExposedService::ClientGateway, adapter, publisher }],
    &listeners,
    shutdown.clone(),
    Backoff::default(),
);
```

Feature `testing` exposes `testing::FakeAdapter`, a scripted adapter with no network and no child processes.
