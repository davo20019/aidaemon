# Exposure adapters: design and build plan

2026-11-12. Status: proposed, nothing built.

## Goal

Let the daemon publish its loopback gateways through a tunnel it supervises,
and feed the live public origin to whoever needs it, so the owner never copies
a URL by hand and never runs a tunnel agent under launchd/systemd themselves.

Non-goals: TLS termination in the daemon, any non-loopback bind, any
authentication in the tunnel layer, exposing the dashboard/health port.

## What exists today

| Piece | Where | State |
|---|---|---|
| Client Gateway | `crates/aidaemon-client-gateway`, loopback `:8788`, routes only under `/client/v1` (`server.rs:44`) | `public_url: Option<String>` is static config |
| Node Gateway | `crates/aidaemon-nodes`, loopback `:8787`, routes only under `/node/v1` (`gateway.rs:166`) | `advertised_endpoints: Vec<String>` is static config, max 3 HTTPS origins (`src/config.rs:2123`) |
| Endpoint rollover for nodes | `HeartbeatResponse.gateway_endpoints` (`crates/aidaemon-nodes/src/service.rs:397`) | Already sent on every heartbeat, but cloned from config |
| Pairing link, in daemon | `/pair_device` uses `gateway.public_url()` (`src/startup/client_gateway.rs:277`) | Runtime object, easy to make live |
| Pairing link, from CLI | `aidaemon web` builds the link from the config file (`src/lib.rs:871`) | Separate process: cannot see daemon runtime state |
| Telegram webhooks | `public_url` per bot, cloudflared + wrangler flow in `src/wizard.rs` | A third consumer of a tunnel, set up by the wizard, run by hand |
| Agent setup | `manage_config enable_client_gateway` (`src/tools/config_manager.rs:1559`), kernel-approved, restart to apply | The model to copy |

Both gateways already own their listener and mount nothing outside their
prefix. That is the property the security argument rests on, so step 1 pins it
with a test instead of trusting adapters to filter paths.

## Invariants

1. **Transport, never authentication.** An adapter moves bytes. Device keys,
   signed challenges, scopes and risk ceilings stay in the protocols. No
   adapter result is ever an input to an auth decision.
2. **Loopback only.** Adapters receive a loopback `SocketAddr`. The existing
   bind checks stay as they are.
3. **One listener, one service.** An exposed port serves exactly one service
   prefix. The supervisor refuses to expose any address that is not a
   registered service listener (so never `8080`). Enforced on the listener,
   because quick tunnels, ngrok and `command` scripts forward a whole port.
4. **Every origin is validated.** Whatever an adapter returns goes through the
   same check `advertised_endpoints` uses today: HTTPS, no credentials, path,
   query or fragment, at most 192 bytes. A bad origin is an adapter failure,
   not a published endpoint.
5. **Unstable origins never reach nodes.** A node that enrolled against an
   origin must be able to find the gateway after a restart. Only adapters that
   declare a stable origin may serve `NodeGateway`.
6. **Exposure is an owner decision.** Turning it on is a kernel-approved
   config effect. The `command` adapter can only be configured by editing the
   file by hand; the agent action cannot write it.
7. **Cross-platform.** No launchd/systemd assumptions. Adapters find their
   binary on `PATH` (or a configured path) and report a clear "not installed"
   health state.

## Crate: `aidaemon-exposure`

Depends on `tokio`, `async-trait`, `serde`, `url`, `tracing`,
`aidaemon-core` (for `CancellationToken`). No dependency on either gateway
crate: the gateways depend on a small endpoint-source type instead (below).

```rust
pub enum ExposedService { ClientGateway, NodeGateway, Webhook { slug: String } }

pub enum OriginStability { Stable, Ephemeral }

pub struct PublicEndpoint { pub origin: Url, pub stability: OriginStability }

pub enum ExposureHealth {
    Starting,
    Up(PublicEndpoint),
    Degraded { endpoint: PublicEndpoint, reason: String },
    Down { reason: String },
}

#[async_trait]
pub trait ExposureAdapter: Send + Sync {
    fn id(&self) -> &str;                         // "cloudflare", "tailscale", ...
    /// Whether this adapter, as configured, yields an origin that survives restarts.
    fn stability(&self) -> OriginStability;
    /// Publish `local` (loopback). The handle owns this one exposure.
    async fn expose(
        &self,
        service: &ExposedService,
        local: SocketAddr,
    ) -> Result<Box<dyn Exposure>, ExposureError>;
}

#[async_trait]
pub trait Exposure: Send + Sync {
    /// Current state; changes whenever the origin or health changes.
    fn watch(&self) -> tokio::sync::watch::Receiver<ExposureHealth>;
    async fn stop(self: Box<Self>) -> Result<(), ExposureError>;
}
```

Why this shape, against the first sketch (`start`/`health`/`stop` on the
adapter):

- One adapter instance can serve two services; a per-exposure handle makes
  `stop` and health unambiguous.
- Origins change (quick tunnels, restarts). A `watch` channel carries that;
  a single return value cannot.
- `stability()` lets the supervisor enforce invariant 5 before starting
  anything.

### Supervisor

`ExposureSupervisor::spawn(config, services, shutdown)`:

- Takes the registered service listeners (`ExposedService` -> loopback addr)
  from startup, after the gateways bind.
- Per configured service: build the adapter, check stability against the
  service, call `expose`, forward the handle's `watch` into that service's
  `EndpointSource`.
- Restart with capped exponential backoff (1 s -> 60 s, reset after 5 min up).
  While down, the `EndpointSource` falls back to the static config value and
  status says so.
- On shutdown: `stop()` every handle, then kill any child still alive.
  Children are spawned with kill-on-drop so a daemon crash does not leak them.
- Emits one event per state change (for `/exposure` and logs). No secrets or
  full child stderr in events.

### `EndpointSource`

Lives in `aidaemon-core` (decision 2): both gateway crates already depend on
it and it already has `tokio` `sync`. The gateways hold it instead of a config
string and never depend on `aidaemon-exposure`.

```rust
#[derive(Clone)]
pub struct EndpointSource { /* watch::Receiver<Option<String>> + Vec<String> fallback */ }
pub struct EndpointPublisher { /* watch::Sender<Option<String>> */ }

impl EndpointSource {
    pub fn fixed(origins: Vec<String>) -> Self;                 // today's behaviour; tests
    pub fn live(fallback: Vec<String>) -> (Self, EndpointPublisher);
    pub fn current(&self) -> Vec<String>;   // live origin first, then fallback, deduplicated
}
impl EndpointPublisher {
    pub fn set(&self, origin: Option<String>);                  // None while down
}
```

Origins are plain strings, as `advertised_endpoints` and the protocol field
`gateway_endpoints` are today, so `aidaemon-core` gains no `url` dependency.
Validation is one function, `aidaemon_exposure::validate_https_origin`, moved
out of `src/config.rs`; config loading and the supervisor both call it, and
the supervisor is the only caller of `EndpointPublisher::set`.

Consumers:

- **Node Gateway**: `NodeService` reads `EndpointSource::current()` for
  `gateway_endpoints` in the heartbeat and in enrollment/bootstrap. Order:
  live origin, then `advertised_endpoints` from config, truncated to 3.
- **Client Gateway**: `ClientGateway::public_url()` returns the first entry of
  `current()`; with none it falls back to the loopback address as today. The
  fallback list is `config.public_url` when set.
- **`aidaemon web` CLI**: stop deriving the link from the config file. Add
  `public_url` to the `PairingOffer` response so the daemon, which knows the
  live origin, supplies it. The field is optional on the wire; the CLI uses
  the config value only when it is absent (older daemon).

## Adapters

| Adapter | Stability | How | Notes |
|---|---|---|---|
| `manual` | Stable | Returns the configured `url`; health is a periodic HTTPS HEAD to the service prefix | Covers a self-run reverse proxy. Replaces nothing, but gives one config shape and health reporting |
| `cloudflare` (named) | Stable | Runs `cloudflared tunnel run` for a configured tunnel + hostname | Replaces the hand-run launchd agent. Reuse the probes in `src/wizard.rs` |
| `cloudflare` (quick) | Ephemeral | `cloudflared tunnel --url`, parse the `trycloudflare.com` origin | Client Gateway only; for testing |
| `tailscale` (serve) | Stable | `tailscale serve --https=443 <local>` in the foreground as a supervised child (never `--bg`, decision 4); origin from `tailscale status --json` (`Self.DNSName`) | Private to the tailnet. Recommended default for new users |
| `tailscale` (funnel) | Stable | Same with `funnel` | Public; flagged as such on the approval card |
| `command` | Declared in config (`stable = true/false`, default false) | Spawns `command` with `{port}`, `{addr}`, `{service}` substituted; reads one JSON line `{"url": "https://..."}` from stdout within a timeout; child must stay alive | Lets a contributor wrap frp, bore or anything else without Rust. Hand-configured only |

`ngrok` is not planned. `command` covers it until someone asks.

Process adapters share one helper: spawn with a scrubbed environment, capture
stderr to a bounded ring (for the `Down { reason }`), kill-on-drop, and never
pass daemon secrets. Adapter credentials (tunnel tokens) are resolved through
`aidaemon-secrets`, never read from the config file in plain text.

## Config

```toml
[exposure.client_gateway]
adapter = "tailscale"            # manual | cloudflare | tailscale | command
# mode = "serve"                 # tailscale: serve | funnel; cloudflare: named | quick
# hostname = "app.example.com"   # cloudflare named
# tunnel = "aidaemon"            # cloudflare named
# url = "https://app.example.com"            # manual
# command = ["my-tunnel", "--port", "{port}"]  # command
# stable = false                 # command

[exposure.node_gateway]
adapter = "cloudflare"
mode = "named"
hostname = "nodes.example.com"
tunnel = "aidaemon"
```

Validation at load (`src/config.rs`, next to the gateway checks):

- An `[exposure.X]` table requires that gateway to be enabled.
- `node_gateway` with an ephemeral adapter is refused with a message naming
  the stable options.
- `manual.url` and `cloudflare.hostname` pass the HTTPS-origin check.
- `deny_unknown_fields` on every table.

`public_url` and `advertised_endpoints` stay and keep working; with no
`[exposure]` table nothing changes.

## Agent and owner surface

- `manage_config enable_exposure` with `service` and `adapter` (enum:
  `manual`, `cloudflare`, `tailscale` only) plus that adapter's fields.
  Same pattern as `enable_client_gateway`: `require_kernel_approval`, backup,
  validate, write, owner-only permissions, "restart to apply".
- The approval card states the service, the adapter and whether the result is
  public or tailnet-only. It is a `Configuration` effect on the config path
  like the other writes; if the kernel grows a typed network-exposure effect
  class, declare that too (do not key anything on the action name).
- `command` is rejected by the action with a message pointing at the config
  file. A generic `manage_config set exposure.*.command` must be refused the
  same way; add it to the sensitive-key list.
- `/exposure`: per service, adapter, state, current origin, time since last
  change, last failure reason. Read-only, no approval.

## Telegram webhooks

Same problem, third consumer. Out of scope for the first three steps, but the
`ExposedService::Webhook` variant is in the enum from day one so the trait
does not need to change. Step 5 moves the wizard's cloudflared ingress onto
the supervisor. Until then the wizard flow is untouched.

## Build order

Each step is one commit series that passes the pre-commit checklist alone.

1. **Crate and plumbing, no behaviour change.**
   - `aidaemon-core`: `EndpointSource` / `EndpointPublisher`.
   - `aidaemon-exposure`: types, traits, supervisor, a `FakeAdapter` for
     tests, `validate_https_origin` (moved out of `src/config.rs`, which now
     calls it).
   - Tests: supervisor restart/backoff with the fake; ephemeral adapter
     refused for `NodeGateway`; invalid origin treated as failure; shutdown
     stops handles.
   - Route-prefix tests in both gateway crates: every mounted route starts
     with `/client/v1` or `/node/v1`.
2. **Gateways read `EndpointSource`.**
   - `NodeService` heartbeat and enrollment; `ClientGateway::public_url()`;
     `public_url` on `PairingOffer`; `aidaemon web` uses it.
   - Startup builds `EndpointSource::fixed` from config. Still no adapter in
     the daemon; existing tests must pass unchanged.
3. **`cloudflare` named adapter + config + `/exposure`.**
   - `[exposure.*]` parsing and validation; supervisor wired in `src/core.rs`
     after both gateway starts, on the daemon `shutdown` token.
   - Live test: remove the launchd cloudflared agent, restart the daemon,
     pair a browser through the tunnel, confirm a node rolls over on
     heartbeat. This is the point where the feature is real.
4. **`tailscale`, `manual`, `command`, `cloudflare` quick; `enable_exposure`.**
   - `command` contract test with a shell fixture that prints the JSON line.
   - `crates/EXTENDING.md`: an "exposure adapter" row and a short section;
     README section for owners.
5. **Telegram webhooks onto the supervisor** (separate decision; only if the
   wizard flow is still a pain point by then).

## Testing

- Unit: everything in step 1 runs on the fake adapter, no network, no child
  processes.
- Process helper: tested with a stub script on Unix and a `cmd` stub on
  Windows, gated by `cfg(target_os)`.
- Real adapters: parsing of captured `cloudflared` / `tailscale status --json`
  output from `.golden` fixtures (not `.txt`/`.json`, which are gitignored).
  No CI job talks to Cloudflare or Tailscale.
- Fixtures use `example.com` hostnames and synthetic tunnel names only.

## Decisions

These were open; they are settled for the implementation.

1. **Restart to apply; no hot apply.** `[exposure.*]` is read once at
   startup, like `[client_gateway]`. `enable_exposure` ends with the same
   "Restart aidaemon (`/restart`) to apply" line as `enable_client_gateway`.
   The supervisor takes its config as a value and owns no global state, so
   hot apply can be added later without changing the trait; do not build it
   now.
2. **`EndpointSource` lives in `aidaemon-core`**, with string origins and no
   `url` dependency. `aidaemon-exposure` depends on core and holds the
   validation function and everything that spawns processes. Neither gateway
   crate depends on `aidaemon-exposure`; only the daemon (`src/`) does.
3. **No new `EffectClass`.** `enable_exposure` goes through
   `ConfigManagerTool::effects_for` unchanged: `CallEffects::unknown()` plus
   the `Configuration` key on the config path, which already forces a kernel
   approval. The approval summary carries the service, the adapter and one of
   "public on the internet" / "private to your tailnet" / "wherever your
   proxy publishes it". Add a typed exposure class only when a second tool
   publishes a listener.
4. **Tunnels live and die with the daemon.** Every process adapter runs its
   program as a foreground child with kill-on-drop; nothing is installed as a
   service and nothing is left running in another daemon. That is why
   `tailscale serve` runs without `--bg`: a `--bg` rule would persist in
   `tailscaled` after aidaemon exits and keep publishing a dead port. On
   `stop()` the tailscale adapter also runs `tailscale serve --https=443 off`
   as a best-effort cleanup. Gateways never wait for a tunnel: they start
   first, the supervisor starts after, and until an adapter reports `Up` the
   `EndpointSource` serves the static config origins. With a named tunnel
   those are the same origin, so nodes see no gap beyond the tunnel's own
   connect time.

## Implementation notes for whoever builds this

- Work on PR 14's branch, one commit per build-order step, each passing
  `cargo fmt`, `cargo clippy --all-features -- -D warnings` and the targeted
  tests (`cargo test -p aidaemon-exposure`, `-p aidaemon-nodes`,
  `-p aidaemon-client-gateway`, `--lib config`). Other sessions commit in
  this tree: commit small and often, never clean or reset.
- Add `crates/aidaemon-exposure` to the workspace `members` list in the root
  `Cargo.toml` (it is an explicit list, not a glob), with a `README.md` and
  version `0.2.0` like its siblings, and an entry in `crates/CHANGELOG.md`.
- Supervisor wiring goes in `src/startup/exposure.rs`, called from
  `src/core.rs` after `startup::client_gateway::start` and the node gateway
  start, with the daemon `shutdown` token. The two gateway `start` functions
  take an `EndpointSource` argument; `core.rs` creates the
  `(EndpointSource, EndpointPublisher)` pairs and hands the publishers to the
  supervisor.
- `/exposure` is a command like `/pair_device` (`DeviceCommands` in
  `src/startup/client_gateway.rs` is the template).
- Refuse `manage_config set` on any `exposure.*.command` key, and on
  `exposure.*.adapter = "command"`, by config path (`ConfigPath`), not by
  matching text.
- Do not touch `src/wizard.rs` or Telegram webhook code before step 5.
- `command` adapter contract: argv only (no shell), placeholders `{port}`,
  `{addr}`, `{service}` (`client_gateway` / `node_gateway`); first stdout
  line must be `{"url": "https://..."}` within 30 s; later stdout is ignored;
  child exit means `Down` and a supervised restart.
- Unknown-binary and not-logged-in cases are `Down { reason }` with the
  install or login command in the reason; they back off, they never crash
  startup.
