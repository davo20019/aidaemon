# aidaemon

[![CI](https://github.com/davo20019/aidaemon/actions/workflows/ci.yml/badge.svg)](https://github.com/davo20019/aidaemon/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/aidaemon)](https://crates.io/crates/aidaemon)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[Website](https://aidaemon.ai/) · [Documentation](https://docs.aidaemon.ai/) · [Discord](https://discord.gg/JCCPtEEy) · [𝕏](https://x.com/aidaemon_ai)

A self-hosted AI agent that runs on your machine and talks to you on Telegram, Slack, or Discord. It can run commands, browse the web, manage files, remember things across conversations, and learn your workflows over time.

I built this because I wanted to control my computer from my phone, from anywhere. Send a message on Telegram, and it runs a command, checks a log, deploys something, or researches a topic — then replies with the result. I also wanted it to run on cheap hardware: a Raspberry Pi, an old laptop, a $5/month VPS.

## What makes it different

- **It's yours** — runs on your machine, your API keys, your data. Nothing goes to a third party beyond the LLM API calls you choose to make.
- **It remembers you** — persistent memory with fact extraction, vector embeddings, and semantic recall. It learns your preferences, your projects, the people you mention. Not a shallow "memory" feature — a full knowledge graph that grows over time.
- **It's always on** — runs as a background daemon. Message it on Telegram at 3am from your phone. Schedule tasks with cron syntax or natural language. It's not an app you open — it's an assistant that's always available.
- **It can act** — 40+ tools: terminal commands, file operations, git, web browsing, web research, HTTP requests with OAuth, MCP servers. It doesn't just answer questions — it does things.
- **It's a single binary** — no Docker, no Node.js, no Python environment. One binary, copy it anywhere and run it. Starts in milliseconds, uses minimal RAM.
- **It works with any LLM** — OpenAI, xAI (Grok), Anthropic, Google Gemini, DeepSeek, Moonshot, MiniMax, Ollama, llama.cpp, OpenRouter, Cloudflare AI Gateway, or any OpenAI-compatible API. Switch providers without changing anything else.

## Quick Start

### One-line install

```bash
curl -sSfL https://get.aidaemon.ai | bash
```

### macOS download

Get `AIdaemon-macos-aarch64.dmg` (Apple silicon) or `AIdaemon-macos-x86_64.dmg` (Intel) from the [latest release](https://github.com/davo20019/aidaemon/releases/latest), drag AIDaemon to Applications and open it. No terminal needed. Until the download is notarized, macOS asks once: open it, then **System Settings → Privacy & Security → Open Anyway**.

### Windows download

Get `aidaemon-windows-x86_64.zip` from the [latest release](https://github.com/davo20019/aidaemon/releases/latest), extract it to a folder you keep (for example `%LOCALAPPDATA%\AIDaemon`) and open `aidaemon-app.exe`. Windows SmartScreen may warn about an unrecognized app: **More info → Run anyway**.

### Homebrew

```bash
brew install davo20019/tap/aidaemon
```

### Cargo

```bash
cargo install --locked aidaemon
```

This installs the daemon only; the app comes with the downloads above.

Then open the AIDaemon app (`aidaemon app`, or `aidaemon-app` from a release). The first time it opens, the desktop app starts the daemon on this computer right away, now and at login, with no administrator password (**Run it on another machine instead** pairs with a server; `aidaemon service uninstall` stops it, and the app then asks before starting it again). It finds a model already on the computer (a ChatGPT sign-in, a provider key in the environment, Ollama or llama.cpp running locally) and offers it in one click, or connects one you pick; each is tested before anything is saved, and each provider says whether it uses your plan, bills per token or runs on your own machine. Self-hosted models need no key: pick **Ollama (local)** or **llama.cpp (local)**, or **Another OpenAI-compatible service** with your server's address (LM Studio, vLLM) and the key left empty. `llama-server` and this daemon's health check both default to port 8080: start one of them elsewhere (`llama-server --port 8081`, or `[daemon] health_port`) and give llama.cpp's address in its form. Your first conversation opens next. To talk to your agent from Telegram, Slack or Discord later, use **Settings → Reach it from your phone**.

On a server, run `aidaemon` there with no config: it prints a link and a one-time code. Reach the machine's port 8788 through an SSH tunnel or your own HTTPS tunnel, open the link on your laptop or phone, and set it up from there (the code pairs the first device only). Prefer a terminal? `aidaemon setup` connects the model right there.

## Features

### Channels
Talk to it on **Telegram**, **Slack** (Socket Mode), or **Discord**. Run multiple bots, add new ones at runtime with `/connect`, no restart needed.
Send images with a caption and the agent analyzes them via vision-capable LLMs (toggle with `[files] vision_enabled` in config).
Ask it to create an image and the provider-neutral `generate_image` tool saves and attaches the result, ready for `send_file`. The bundled image backend uses a connected ChatGPT subscription and works even when the conversational model is Anthropic, Gemini, or another configured provider.
Send voice notes or audio files on Telegram, Slack, or Discord. Audio-capable models receive native `input_audio` blocks; otherwise enable local Whisper STT fallback under `[files.stt]` (requires `whisper-cli`, a GGUF model, and `ffmpeg` for OGG/Opus). See `config.toml.example` for settings. Restart after config changes.
Telegram owners can also run `/terminal [agent] [working_dir]` to launch the hosted terminal Mini App (`https://terminal.aidaemon.ai`) for direct CLI-agent sessions with automatic secure daemon pairing.

### Tools & Agents
40+ built-in tools including terminal execution, file operations, git, web browser (headless Chrome), web search, HTTP requests with auth profiles, and MCP server integration. Tools declare their risk level — read-only operations run freely, side effects require approval.

Delegate complex tasks to **sub-agents** or external **CLI agents** (Claude, Gemini, Codex, Aider, Antigravity — auto-discovered). Any other agent that speaks the [Agent Client Protocol](https://agentclientprotocol.com) works too: add it under `[cli_agents.tools.<name>]` with `protocol = "acp"`, and each action it takes is asked first, so every one can be judged.

`manage_api` onboards an API from its OpenAPI spec or GraphQL schema and derives a safe verification probe from it (GraphQL reuses introspection when no `verify_url` is given). Credentials stay in the OS keychain by default, or in the configured `.env` file when `AIDAEMON_NO_KEYCHAIN=1`.

### Computer use (macOS)
Built with `--features computer_use-macos`, the agent can operate native macOS apps: it reads an app's accessibility tree, takes a screenshot, and clicks, types and scrolls to finish a task. It is off by default: build with the feature and enable it in config. Setup and permissions: [COMPUTER_USE_MACOS.md](COMPUTER_USE_MACOS.md).

### Memory
SQLite-backed persistent memory with local vector embeddings (quantized AllMiniLM-L6-v2 by default; any fastembed model via `[memory] embedding_model`). Background consolidation extracts durable facts from conversations. Semantic episode recall is the coarse discovery layer; `search_history` adds authorized FTS5 search over exact retained user/assistant messages, anchored context, task bookends, and signed forward/backward paging. Exact recall follows `state.retention.messages_days` (90 by default; `0` opts into permanent active-database history), independently of short-lived diagnostic events. Database encrypted at rest by default.

**Project memory.** A conversation placed in a project (the app's Code conversations, or a chat moved there with `/project`) keeps what it learns about that project in the project's own notes: plain Markdown files under `~/.aidaemon/projects/<project-id>/memory/` with a generated `MEMORY.md` index. Notes are outside the repository, keyed by the project's id so a rename or a moved folder keeps them, and you can read or edit them by hand. `remember_fact` takes `scope: "project" | "personal"` (default `project` in a project conversation), and background extraction, consolidation and delegated-task learning route the same way. Facts about you and the people around you always stay in personal memory, and nothing a project learns can replace or merge with them. Each project turn shows the project's index after your own memory; `/memory` shows where it lives. Turn it off with `[memory] project_memory = false`, or move it with `[memory] projects_dir`.

A project conversation also gets the repository's own skills (`.agents/skills`, `.claude/skills`) and can start the MCP servers its `.mcp.json` declares with `manage_mcp project_start`, each only after you approve its exact command; those tools stay in that project's conversations.

### Skills
Instructions that teach it new workflows, loaded when a task needs them or when you ask by name. Load from files, URLs, or remote registries. Successful procedures auto-promote to skills after enough consistent use.

### Automation
Everything that runs on its own is an automation, of two kinds. **Scheduled** automations run a fixed instruction at set times: reminders and recurring tasks, with natural language time parsing and multi-schedule support. **Watching** automations work toward an objective within limits you approve, deciding each time whether to act, wait, ask you, or stop. Ask for what you want; the agent picks the kind. Your automations show in one list, each as the same five-line card (what, how often, what it can do, when it tells you, when it ends), and you pause, resume or cancel any of them by name. Email triggers via IMAP IDLE.

### Security
Terminal command allowlists, inline approval flow (Allow Once / Allow Session / Allow Always / Deny), SSRF protection, input sanitization, encrypted state, secrets in OS keychain, and role-based access control. See the [security docs](https://docs.aidaemon.ai/) for the full model.

## Running as a Service

```bash
aidaemon service install      # this user, at login, from ~/.aidaemon (--dir to change); starts it now
aidaemon service uninstall
```

This is what the app's **Run it here** does: a launchd agent on macOS, a systemd user unit on Linux (lingering is turned on so it outlives your login session), a logon task on Windows. `aidaemon install-service` still writes the system-wide unit (Linux, needs sudo) or the launch agent without starting it (macOS).

## Configuration

All settings live in `config.toml`, written by first-run setup and editable from the app's Settings page. See [`config.toml.example`](config.toml.example) for the full reference and the [documentation](https://docs.aidaemon.ai/) for detailed guides on each feature. The provider config supports both model-level fallbacks on the primary backend and ordered cross-provider fallbacks via `[[provider.fallbacks]]`.

### Execution environments

By default, agent-visible files and commands run locally with the same behavior
as earlier releases. You can instead put the entire workspace behind a
persistent Docker container or an SSH host:

```toml
[execution]
backend = "local" # local | docker | ssh
# workspace_root = "/absolute/path/to/workspace"
# allow_outside_workspace = true # path tools: true locally, false remotely
```

Filesystem tools, terminal commands, Git operations, CLI agents, attachments,
and generated result files all use the selected backend. This keeps the model
on one coherent filesystem. Configuration, encrypted state, channel transport,
credentials, approvals, and browser/computer-use sessions remain on the daemon
host.

The Docker backend is built in (the `execution-docker` feature, on by
default; it drives the `docker` CLI and adds no dependencies), so Docker is
needed only when `[execution] backend = "docker"` is set. The Docker backend creates or reuses one persistent container and defaults to
no network access. The SSH backend uses the local OpenSSH client and honors
existing host aliases, ssh-agent, ProxyJump, known-host, and hardware-key
configuration. `allow_outside_workspace` confines path-based tools and command
working directories; shell commands still have the permissions of the
selected local/SSH account or container and remain subject to terminal
approval. Persistent terminal approvals are scoped to the concrete execution
target. See [`config.toml.example`](config.toml.example) for complete Docker and
SSH examples.

### Workspace source checkpoints

For remotely operated agents, the local execution backend can create a
conflict-aware source checkpoint before the first workspace mutation in each
turn:

```toml
[execution]
backend = "local"
workspace_root = "/absolute/path/to/project"
allow_outside_workspace = false

[checkpoints]
enabled = true
```

Checkpoints use a bare shadow-Git object store in aidaemon's platform data
directory. They never modify the project's `.git` directory, index, or
working-tree metadata. `.env*` secrets (except example/template files), keys,
credentials, dependencies, build outputs, caches, logs, special files, and
oversized files are excluded; project `.gitignore` rules are respected.

Owners can use `/checkpoints` or the read-only `list_checkpoints` tool to list
them. `/rollback <id>` produces a conflict preview and short-lived confirmation
token; `/rollback confirm <token>` restores only paths still matching the
agent's finalized post-turn tree. Independently changed paths are preserved,
and a safety checkpoint is taken before rollback.

This beta is local-backend only. Raw terminal/agent bridge sessions are not
covered, and checkpoints involving detached/background mutation are marked
unsafe rather than offered for rollback. The external object store is
owner-readable and contains plaintext copies of eligible source files; use
retention and size limits appropriate for the machine.

### Authority and untrusted content

Pages, emails, other people's messages and files outside your own folders can
contain text written to steer an AI agent. aidaemon tracks this. Once a turn
has read content someone else wrote, any outward action it then tries (sending
a message or email, posting, calling a remote API with your credentials,
deleting outside your folders, changing configuration, running an unknown
program on the host) is first checked by a separate model call that sees only
your request and the proposed action, never the content that was read. If the
action is not something your request asked for, it is refused and the agent
is told why. Reading, editing files in your workspace and ordinary local work
such as builds, tests and local git commits are not checked. Results from
helper agents carry the same mark, and a chat stays marked while the turn that
read the content is still in its context.

```toml
[authority]
owner_default = "full"   # full | unrestricted
# Your own folders: reading files here does not count as untrusted content.
trusted_roots = ["~/projects"]
```

Your own folders are only the ones you grant on purpose: `[authority]
trusted_roots`, the daemon's working directory, a scheduled goal's workspace
set with `bind_workspace`, and a group chat's workspace grant. Path aliases
(`[path_aliases] projects`) only help aidaemon find projects by name; they do
not grant trust, so add a folder to `trusted_roots` if you want it trusted.
A folder named only in a message or a goal's mission (for example "summarize
~/Downloads/invoice.pdf") is not trusted: reading it counts as untrusted
content, so the next outward action is checked. The broadest folders never
count, even when configured or reached through a symlink: `/`, your home
folder and its parents, and the shared temp folders. A symlink counts as the
folder it points to. At startup the daemon logs the trusted roots it uses
and any it dropped as too broad.

With `owner_default = "full"` (the default) the check above applies to your
own turns too. `unrestricted` skips the check on the owner's turns and only
records each outward action taken after untrusted content; guests' turns are
always checked.

Chat commands (owner only) change one chat for four hours:

- `/readonly` makes the chat read-only, including work already running and
  the helper agents it started: aidaemon can look but not change anything.
- `/mode full | unrestricted | readonly` switches the chat's mode; `/mode` on
  its own shows the current one. `/mode full` ends a read-only or unrestricted
  period early.

To keep one message out of memory, send `/private` first (anyone can):
aidaemon still uses what it knows to answer, but learns nothing from that
message, now or later. `/private off` cancels. You can also just ask it not
to remember something; the agent then marks that turn the same way.

### Approvals

Some actions wait for your OK. aidaemon sends you a card in the chat that
says what the agent wants to do, how risky it is, and anything worth
knowing. You are asked when the agent wants to:

- send a web request that changes something (anything other than a plain
  GET, HEAD or OPTIONS), including a live test of a new API connection;
- run a terminal command that is not on your allowlist, unless aidaemon's
  safety check looks at it and finds nothing risky. If the check flags it,
  fails, takes too long or is not set up, you are asked;
- run a terminal command that needs a look whatever the allowlist says:
  reading sensitive files such as `~/.ssh`, starting a program that keeps
  running in the background, running a new command detached from the task,
  reaching outside the task's folders in a sandboxed terminal, or turning on
  trust-all mode;
- run any terminal command in a chat started by an automated trigger, such as
  an incoming email.

Trusted sessions run their web requests and terminal commands without
cards: scheduled goals and tasks you set up, and automated events marked as
trusted. In exchange, they cannot start programs that keep running in the
background. One exception: terminal commands in a chat started by an
automated trigger are always asked about, trusted or not. An Autopilot
mandate's web requests are not asked about either: they run only when they
match the action you granted the mandate. Before anything is asked, outgoing
requests are checked for secrets such as API keys and tokens: a request
carrying one is blocked, not asked about.

How to answer:

- **Allow Once** runs this one action.
- **Allow Session** also allows the same action again without asking, until
  the daemon restarts. For a web request this covers the same method to the
  same address, from this chat and the helper agents it starts. For a terminal command it covers
  commands that start the same way, from any chat.
- **Allow Always** (terminal commands) saves the command's start to your
  allowlist, so commands like it run without a card from now on.
- **Deny** stops the action.

Web request cards offer Allow Session. Terminal cards offer either Allow
Session or Allow Always, depending on the terminal's permission mode and the
command's risk (by default, Allow Session for critical commands and Allow
Always for the rest). A card waits five minutes; no answer, or a card that
could not be delivered, counts as Deny.

A Deny holds for the rest of the task. The agent cannot get around it by
trying another way: once you decline a POST to a site, a `curl` to that site
from the terminal is refused too, and so is the same request from any helper
agent the task starts. No new card is sent for these retries. If, after a
Deny, the agent tries something that sends data out or deletes, and whose
effects cannot be worked out in advance (an unfamiliar script, for example),
you are asked again; it is never let through quietly. A task in which you
declined something is recorded as only partly done, never as a success. A
message you send while the task is still running joins that task, and the
Deny still holds. Once the task has finished, your next message starts a new
task, so you can change your mind and ask again.

### Reaching the gateways from your other devices

The Client Gateway (`/client/v1`, for `aidaemon tui`, `aidaemon web` and paired phones) and the Node Gateway (`/node/v1`) only listen on loopback. To reach them from elsewhere, let the daemon run the tunnel itself: it starts with the daemon, stops with it, restarts with backoff when it drops, and feeds its address into pairing links and Node heartbeats. The tunnel only moves bytes; devices still pair with their own keys.

```toml
[exposure.client_gateway]
adapter = "tailscale"        # private to your tailnet; mode = "funnel" makes it public

[exposure.node_gateway]
adapter = "cloudflare"       # a named tunnel on your own hostname
hostname = "nodes.example.com"
tunnel = "my-tunnel"         # created with `cloudflared tunnel create`
```

Adapters: `tailscale` (`serve` or `funnel`), `cloudflare` (`named`, or `quick` for a throwaway `trycloudflare.com` address, Client Gateway only), `manual` (`url` of a reverse proxy you run) and `command` (any program that prints `{"url": "https://..."}` on its first line and keeps running; argv only, `{port}`, `{addr}` and `{service}` are filled in). The Node Gateway needs an address that survives restarts, so it refuses `quick` and a `command` without `stable = true`. `cloudflared` or `tailscale` must be on `PATH` (or set `binary`). Changes apply after a restart; `/exposure` shows each tunnel's state, address and last failure. The agent can set this up for you (`manage_config enable_exposure`, with your approval), except `command`, which only you can write.

### The phone app (iOS and Android)

The app on a phone never runs the daemon; it pairs with yours. Expose the Client Gateway first (above). The quickest way: in the desktop app on the daemon's computer, open Settings → Paired devices → **Pair a phone**, scan the QR code with the phone's camera, open it in AIDaemon, and approve the phone on the desktop. Or send `/pair_device` to your agent from the desktop app, Telegram, Slack or Discord, paste the link it replies with into the phone app (an `aidaemon-pair:` link or QR code opens the app and pairs at once), and approve the phone in that chat. The phone keeps its key in the iOS keychain or Android's private app storage; `/devices` lists it and `/revoke_device` unpairs it, after which the app offers to pair again.

Build it from `clients/app` with the [nsh](https://github.com/longbridge/nsh) CLI: `nsh package --target android` (an `.apk`; `--keystore` to sign for release, `--format aab` for Google Play) or `nsh package --target ios --sign "<identity>" --profile <profile>.mobileprovision` (needs Xcode). `nsh run --target ios` or `--target android` runs it in a simulator or on a device.

### Customize your agent

Your agent's first message offers to change its name and personality: just tell it. It writes them to an editable `persona.md` next to `config.toml`.

You can also configure it manually:

```toml
[agent]
name = "Nova"
persona_file = "persona.md"
```

Use the persona file for the agent's role, communication style, priorities, and working preferences:

```markdown
# Role

You are a research and project-management partner.

# Communication style

- Be concise and direct.
- Challenge questionable assumptions constructively.
- Explain technical subjects in plain language.
```

Relative persona paths resolve next to `config.toml`; absolute paths are also supported. Restart aidaemon after changing the name, persona path, or persona file. Persona instructions customize behavior without replacing aidaemon's privacy, security, tool-use, or completion-honesty rules.

The built-in policy profiles `cheap`, `balanced`, and `strong` are execution presets, not separate model tiers. They all use your configured default model; the differences are context budget, tool budget, verification depth, and approval behavior. The open-source default auto-routing now floors at `balanced` for better reliability, while `cheap` remains available as a lower-budget preset for custom forks and policy tweaks.

Model supervision is trust-tiered. By default (`trust_tier = "auto"` under `[policy]`), recognized frontier models (Claude, GPT-4/5, o-series, Gemini 2.5+, Grok 3/4) run a thin loop where behavioral supervision gates are telemetry-only, while every other model gets the full supervision scaffolding. The auto list matches known model families, so a capable open model it doesn't recognize (for example kimi-k2.5 or a large qwen/deepseek variant) lands in the supervised tier, where a gate misfire can cost extra LLM round-trips before an answer is accepted. If you run such a model, set `trust_tier = "autonomous"` to opt out of the supervision gates; anti-fabrication checks and hard safety caps (iteration/token/wall-clock limits, repetition guards) remain active on every tier. The override is global — it applies to every configured model, including fallbacks — so keep `"auto"` if your fallback chain mixes a strong cloud model with a small local one. See [`config.toml.example`](config.toml.example) for details.

### Using a ChatGPT subscription instead of an API key

aidaemon can run on a ChatGPT Plus/Pro/Business plan rather than metered OpenAI
API billing. Sign in once:

```bash
aidaemon auth login openai          # add --paste for headless/remote installs
aidaemon auth status
aidaemon auth logout openai
```

First-run setup offers the same sign-in (on another device it asks you to paste the address the browser ends on). By hand, point the provider at it:

```toml
[provider]
kind = "openai_chatgpt"
api_key = ""                        # unused; credentials come from the OAuth store

reasoning_effort = "medium"         # none | minimal | low | medium | high | xhigh | max | ultra

[provider.models]
primary = "gpt-5.6-terra"           # sol = flagship, terra = balanced, luna = fast
fast = "gpt-5.6-luna"
smart = "gpt-5.6-sol"
```

Model and reasoning effort are both yours to pick. Effort is passed through
verbatim, so newly added levels work without a client update; each model
supports its own subset and the API rejects ones it doesn't.

Sign-in is OAuth 2.0 + PKCE against `auth.openai.com`, with the redirect landing
on `http://localhost:1455/auth/callback` (the port is fixed by OpenAI's client
registration — use `--paste` when no local browser can reach it). Tokens are
stored in the OS keychain and refreshed automatically. Requests identify
themselves with an `originator: aidaemon` header.

The same login also enables `generate_image` (when `[files] enabled = true`).
It uses ChatGPT's subscription-backed image service and does not require an
`OPENAI_API_KEY`. The image backend is selected independently from the chat
provider, so a conversation running on Anthropic, Gemini, xAI, or a local model
can still generate through the connected ChatGPT account.

Internally, `generate_image` targets a provider-neutral
`ImageGenerationBackend` interface. The ChatGPT subscription adapter is the
first bundled implementation; adding another provider's native image API only
requires a new adapter, without changing the tool schema, agent loop, attachment
handling, or delivery flow. Until such an adapter is configured, image requests
still require the ChatGPT login above. Disable the tool explicitly with
`[tools] disabled = ["generate_image"]` if that fallback is not wanted.

Subscription plans meter usage: when a cap is hit, the provider reports a rate
limit and the configured fallbacks take over. Background goals and heartbeats
pointed at a subscription model will reach that cap faster than interactive use
alone. These credentials do not work with OpenAI's Realtime endpoints, which
still need a Platform API key.

Secrets (API keys, bot tokens) are stored in your OS keychain by default — not in config files.

If you prefer env-only secrets, set `AIDAEMON_NO_KEYCHAIN=1`. In that mode:
- aidaemon reads secrets from environment variables / `.env`
- secret writes (including OAuth token refresh rotation) are persisted back to the env file
- set `AIDAEMON_ENV_FILE=/absolute/path/.env` to control which env file is used (default: `.env` in working directory)

For env-only mode, keep the env file private (`chmod 600`) and out of version control.

For external APIs, the built-in auth flow is:
- `manage_api` for a single deterministic connect + learn + verify flow
- `manage_oauth` for OAuth services, including custom OAuth 2.0 PKCE, authorization-code, and client-credentials providers you register at runtime
- `manage_http_auth` for API keys, bearer tokens, custom headers, basic auth, or OAuth 1.0a credentials
- `manage_skills` with `learn_api` to ingest the official docs or OpenAPI/Swagger URL into a reusable API guide after auth is set up

`learn_api` now prefers OpenAPI/Swagger when available, crawls multiple docs pages on the same host when needed, discovers linked specs from docs, bundles remote OpenAPI `$ref` documents, and can switch to GraphQL schema introspection when a GraphQL endpoint is available.

## Build your own agent (SDK)

The daemon is built on a Rust agent SDK that you can use on its own: an agent loop driven by hooks, tools with risk annotations and approvals, providers, receipts for every tool call, and run state you can pause and resume. Every SDK crate shares one version (0.21.0); the public-contract crates are semver-checked in CI.

The SDK crates are on [crates.io](https://crates.io/crates/aidaemon-sdk):

```toml
[dependencies]
aidaemon-sdk = "0.21"
```

Or follow `master`: `aidaemon-sdk = { git = "https://github.com/davo20019/aidaemon" }`.

```rust
use aidaemon_sdk::prelude::*;

/// Today's weather for a city.
#[tool(read_only)]
async fn forecast(city: String) -> String {
    format!("{city}: 21°C and sunny")
}

let agent = preset("weather")
    .provider_from_env() // ANTHROPIC_API_KEY or OPENAI_API_KEY
    .instructions("Answer in one sentence.")
    .tool(Forecast)
    .build()
    .await?;
let outcome = agent.run("cli:me", "Weather in Lisbon?").await?;
```

Try it without an API key: `cargo run -p aidaemon-sdk --example quickstart --features testing`. Other examples cover sub-agents, approvals, scheduled workers, your own channel, resuming after a crash and a coding agent ([crates/aidaemon-sdk/examples](crates/aidaemon-sdk/examples)). Guide: [docs.aidaemon.ai](https://docs.aidaemon.ai/).

## Building from Source

```bash
cargo build --release                                   # default build
cargo build --release --features computer_use-macos     # add macOS computer use
cargo test                                              # run tests
```

The default build includes Slack, Discord, the headless browser, database encryption, the Docker execution backend, the terminal bridge and OpenTelemetry export (`otel`).

See [CLAUDE.md](CLAUDE.md) for architecture details, module map, and contributor guidance.

## License

[MIT](LICENSE)
