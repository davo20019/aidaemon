# aidaemon

[Website](https://aidaemon.ai/) · [Documentation](https://docs.aidaemon.ai/) · [GitHub](https://github.com/davo20019/aidaemon) · [Discord](https://discord.gg/JCCPtEEy) · [𝕏](https://x.com/aidaemon_ai)

A personal AI agent that runs as a background daemon, accessible via Telegram, Slack, or Discord, with tool use, MCP integration, web research, scheduled tasks, and persistent memory.

I built this because I wanted to control my computer from my phone, from anywhere. I also wanted it to run on cheap hardware - a Raspberry Pi, an old laptop, a $5/month VPS - without eating all the RAM just to sit idle waiting for messages.

## Why Rust?

aidaemon runs 24/7 as a background daemon. It needs to be small, fast, and run on anything:

- **Runs on cheap/old hardware** - a lightweight Rust binary. On a Raspberry Pi or a $5 VPS with 512 MB RAM, it runs comfortably where heavier runtimes won't.
- **Single binary, zero runtime** - one binary, copy it to any machine and run it. Install with `curl -sSfL https://get.aidaemon.ai | bash` or `cargo install aidaemon`.
- **Startup in milliseconds** - restarts after a crash are near-instant, which matters for the auto-recovery retry loop.
- **No garbage collector** - predictable latency. No GC pauses between receiving the LLM response and sending the reply.

If you don't care about resource usage and want more channels (WhatsApp, Signal, iMessage) or a web canvas, check out [OpenClaw](https://openclaw.ai) which does similar things in TypeScript.

## Features

### Channels
- **Telegram interface** - chat with your AI assistant from any device
- **Slack integration** - Socket Mode support with threads, file sharing, and inline approvals
- **Discord integration** - bot with slash commands and thread support
- **Dynamic bot management** - add or list bots at runtime via `/connect` and `/bots` commands, no restart needed
- **Multi-bot support** - run multiple Telegram, Slack, and Discord bots from a single daemon

### LLM Providers
- **Multiple providers** - native Anthropic, Google Gemini, DeepSeek, and OpenAI-compatible (OpenAI, OpenRouter, Ollama, etc.)
- **ExecutionPolicy routing** - risk-based model selection using tool capabilities (read-only, side-effects, high-impact writes), uncertainty scoring, and mid-loop adaptation
- **Token/cost tracking** - per-session and daily usage statistics with optional budget limits

### Tools & Agents
- **40+ tools** - file operations (read, write, edit, search), git info/commit, terminal, system info, web research, browser, HTTP requests, and more
- **Dynamic MCP management** - add, remove, and configure MCP servers at runtime via the `manage_mcp` tool
- **Browser tool** - headless Chrome with screenshot, click, fill, and JS execution
- **Web research** - search (DuckDuckGo/Brave) and fetch tools for internet access
- **HTTP requests** - authenticated API calls with OAuth 1.0a, Bearer, Header, and Basic auth profiles
- **Sub-agent spawning** - recursive agents with configurable depth, iteration limit, and dynamic budget extension
- **CLI agent delegation** - delegate tasks to claude, gemini, codex, aider, copilot (auto-discovered via `which`)
- **Goal tracking** - long-running goals with task breakdown, scheduled runs, blockers, and diagnostic tracing
- **Channel history** - read recent Slack channel messages with time filtering and user resolution
- **Skills system** - trigger-based markdown instructions with dynamic management, remote registries, and auto-promotion from successful procedures
- **Tool capability registry** - each tool declares read_only, external_side_effect, needs_approval, idempotent, high_impact_write for risk-based filtering

### OAuth & API Integration
- **OAuth 2.0 PKCE** - built-in flows for Twitter/X and GitHub, plus custom providers
- **OAuth 1.0a** - legacy API support (e.g., Twitter v1.1)
- **HTTP auth profiles** - pre-configured auth for external APIs (Bearer, Header, Basic, OAuth)
- **Token management** - tokens stored in OS keychain, automatic refresh, connection tracking

### Memory & State
- **Persistent memory** - SQLite-backed conversation history + facts table, with fast in-memory working memory
- **Memory consolidation** - background fact extraction with vector embeddings (AllMiniLML6V2) for semantic recall
- **Evidence-gated learning** - stricter thresholds for auto-promoting procedures to skills (7+ successes, 90%+ success rate)
- **Context window management** - role-based token quotas with sliding window summarization
- **People intelligence** - organic contact management with auto-extracted facts, relationship tracking, and privacy controls
- **Database encryption** - optional SQLCipher AES-256 encryption at rest (feature flag: `--features encryption`)

### Automation
- **Scheduled tasks** - cron-style task scheduling with natural language time parsing
- **HeartbeatCoordinator** - unified background task scheduler with jitter, semaphore-bounded concurrency, and exponential backoff
- **Bounded auto-tuning** - adaptive uncertainty threshold that adjusts based on task failure ratios
- **Email triggers** - IMAP IDLE monitors your inbox and notifies you on new emails
- **Background task registry** - track and cancel long-running tasks

### File Transfer
- **File sharing** - send and receive files through your chat channel
- **Configurable inbox/outbox** - control where files are stored and which directories the agent can access

### Security & Config
- **Config manager** - LLM can read/update `config.toml` with automatic backup, restore, and secrets redaction
- **Command approval flow** - inline keyboard (Allow Once / Allow Always / Deny) for unapproved terminal commands
- **HTTP write approval** - POST/PUT/PATCH/DELETE requests require user approval with risk classification
- **Secrets management** - OS keychain integration + environment variable support for API keys

### Operations
- **Web dashboard** - built-in status page with usage stats, active sessions, and task monitoring
- **Channel commands** - `/model`, `/models`, `/auto`, `/reload`, `/restart`, `/clear`, `/cost`, `/tasks`, `/cancel`, `/connect`, `/bots`, `/help`
- **Auto-retry with backoff** - exponential backoff (5s -> 10s -> 20s -> 40s -> 60s cap) for dispatcher crashes
- **Health endpoint** - HTTP `/health` for monitoring
- **Service installer** - one command to install as a systemd or launchd service
- **Setup wizard** - interactive first-run setup, no manual config editing needed

## Quick Start

### One-line install (any VPS / Linux / macOS)

```bash
curl -sSfL https://get.aidaemon.ai | bash
```

Downloads the latest binary, verifies its SHA256 checksum, and installs to `/usr/local/bin`.

### Homebrew (macOS / Linux)

```bash
brew install davo20019/tap/aidaemon
aidaemon  # launches the setup wizard on first run
```

### Cargo

```bash
cargo install aidaemon
aidaemon
```

### Build from source

```bash
cargo build --release
./target/release/aidaemon
```

The wizard will guide you through:
1. Selecting your LLM provider (OpenAI, OpenRouter, Ollama, Google AI Studio, Anthropic, etc.)
2. Entering your API key
3. Selecting and setting up one or more channels (Telegram, Slack, Discord)

## Configuration

All settings live in `config.toml` (generated by the wizard). See [`config.toml.example`](config.toml.example) for the full reference.

### Secrets Management

API keys and tokens can be specified in three ways (resolution order):

1. **`"keychain"`** — reads from OS credential store (macOS Keychain, Windows Credential Manager, Linux Secret Service)
2. **`"${ENV_VAR}"`** — reads from environment variable (for Docker/CI)
3. **Plain value** — used as-is (not recommended for production)

The setup wizard stores secrets in the OS keychain automatically.

### Provider

```toml
[provider]
kind = "openai_compatible"  # "openai_compatible" (default), "google_genai", or "anthropic"
api_key = "keychain"        # or "${AIDAEMON_API_KEY}" or plain value
base_url = "https://openrouter.ai/api/v1"

[provider.models]
primary = "openai/gpt-4o"
fast = "openai/gpt-4o-mini"
smart = "anthropic/claude-sonnet-4"
```

The `kind` field selects the provider protocol:
- `openai_compatible` (default) — works with OpenAI, OpenRouter, Ollama, DeepSeek, or any OpenAI-compatible API
- `google_genai` — native Google Generative AI API (Gemini models)
- `anthropic` — native Anthropic Messages API (Claude models)

The three model tiers (`fast`, `primary`, `smart`) are used by the smart router. Simple messages (greetings, short lookups) route to `fast`, complex tasks (code, multi-step reasoning) route to `smart`, and everything else goes to `primary`.

### Telegram

```toml
[telegram]
bot_token = "keychain"           # or "${TELOXIDE_TOKEN}" or plain value
allowed_user_ids = [123456789]
```

### Slack
Enabled by default in standard builds:

```toml
[slack]
app_token = "keychain"           # xapp-... Socket Mode token
bot_token = "keychain"           # xoxb-... Bot token
allowed_user_ids = ["U12345678"] # Slack user IDs (strings)
use_threads = true               # Reply in threads (default: true)
```

Slack is activated automatically when both `app_token` and `bot_token` are set.
If you want a minimal binary, build with `--no-default-features` and re-enable only what you need.

### Terminal Tool

```toml
[terminal]
# Set to ["*"] to allow all commands (only if you trust the LLM fully)
allowed_prefixes = ["ls", "cat", "head", "tail", "echo", "date", "whoami", "pwd", "find", "grep"]
```

### MCP Servers

MCP servers can be configured statically or added at runtime via the `manage_mcp` tool.

```toml
[mcp.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
```

The `manage_mcp` tool supports runtime management:
- **add** — add and start a new MCP server (allowed commands: `npx`, `uvx`, `node`, `python`, `python3`)
- **list** — list all registered servers and their tools
- **remove** — remove a server
- **set_env** — store API keys for a server in the OS keychain
- **restart** — restart a server with fresh environment from keychain

### Browser

```toml
[browser]
enabled = true
headless = true
screenshot_width = 1280
screenshot_height = 720
# Use an existing Chrome profile to inherit cookies/sessions
# user_data_dir = "~/Library/Application Support/Google/Chrome"
# profile = "Default"
```

### Sub-agents

```toml
[subagents]
enabled = true
max_depth = 3              # max nesting levels
max_iterations = 10        # initial agentic loop iterations per sub-agent
max_iterations_cap = 25    # max iterations even with dynamic budget extension
max_response_chars = 8000
timeout_secs = 300         # 5 minute timeout per sub-agent
```

Sub-agents can request additional iterations via the `request_more_iterations` tool, up to `max_iterations_cap`.

### CLI Agents

```toml
[cli_agents]
enabled = true
timeout_secs = 600
max_output_chars = 16000

# Tools are auto-discovered via `which`. Override or add your own:
[cli_agents.tools.claude]
command = "claude"
args = ["-p", "--output-format", "json"]

[cli_agents.tools.gemini]
command = "gemini"
args = ["-p", "--output-format", "json", "--sandbox=false"]
```

### Skills

Skills are trigger-based markdown instructions that guide the agent's behavior. They can be loaded from a directory, added from URLs, created inline, or installed from remote registries.

```toml
[skills]
enabled = true
dir = "skills"    # relative to config.toml location

# Optional: remote registries for browsing and installing community skills
registries = [
    "https://example.com/skills/registry.json"
]
```

The `manage_skills` tool supports runtime management:
- **add** — add a skill from a URL
- **add_inline** — create a skill from raw markdown with YAML frontmatter
- **list** — list all loaded skills with their status and triggers
- **remove/enable/disable** — manage individual skills
- **browse** — search remote skill registries
- **install** — install a skill from a registry
- **update** — re-fetch a skill from its source URL

Successful procedures (>= 7 uses, >= 90% success rate) are automatically promoted to skills every 12 hours via evidence-gated learning.

### OAuth

OAuth enables the agent to authenticate with external services like Twitter/X and GitHub. Built-in providers require no URL configuration — just enable OAuth and set credentials.

```toml
[oauth]
enabled = true
callback_url = "http://localhost:8080"  # must match your OAuth app's redirect URI

# Optional: add custom OAuth providers beyond the built-in Twitter/GitHub
[oauth.providers.stripe]
auth_type = "oauth2_pkce"
authorize_url = "https://connect.stripe.com/oauth/authorize"
token_url = "https://connect.stripe.com/oauth/token"
scopes = ["read_write"]
allowed_domains = ["api.stripe.com"]
```

Built-in providers (no URL config needed):
- **twitter** (alias: **x**) — Tweet read/write, user info, offline access
- **github** — User info, repository access

The `manage_oauth` tool handles the full lifecycle:
- **providers** — list available providers and credential status
- **set_credentials** — store client_id/client_secret in OS keychain
- **connect** — start OAuth flow (displays authorize URL, waits for callback)
- **list** — show connected services with token expiry
- **refresh** — refresh an expired access token
- **remove** — disconnect a service

### HTTP Auth Profiles

Pre-configured auth profiles for external APIs, used by the `http_request` tool. Supports four auth types:

```toml
# Bearer token (OAuth 2.0 or API key)
[http_auth.stripe]
auth_type = "bearer"
allowed_domains = ["api.stripe.com"]
token = "keychain"

# OAuth 1.0a (e.g., Twitter v1.1 API)
[http_auth.twitter_v1]
auth_type = "oauth1a"
allowed_domains = ["api.twitter.com"]
api_key = "keychain"
api_secret = "keychain"
access_token = "keychain"
access_token_secret = "keychain"

# Custom header auth
[http_auth.custom_api]
auth_type = "header"
allowed_domains = ["api.example.com"]
header_name = "X-API-Key"
header_value = "keychain"

# Basic auth
[http_auth.internal]
auth_type = "basic"
allowed_domains = ["internal.company.com"]
username = "service_account"
password = "keychain"
```

All credential fields support `"keychain"` for OS keychain storage. The `allowed_domains` field is required and enforces domain restrictions on each profile.

OAuth connections established via `manage_oauth` automatically create auth profiles — no manual config needed for built-in providers.

### People Intelligence

Organic contact management that learns about people from conversations. Disabled by default.

```toml
[people]
enabled = true
auto_extract = true                    # learn facts from conversations
auto_extract_categories = [            # categories to auto-extract
    "birthday", "preference", "interest",
    "work", "family", "important_date"
]
restricted_categories = [              # never auto-extracted
    "health", "finance", "political", "religious"
]
fact_retention_days = 180              # auto-delete stale facts
reconnect_reminder_days = 30           # suggest reconnecting after inactivity
```

The `manage_people` tool provides manual control:
- **add/list/view/update/remove** — manage person records
- **add_fact/remove_fact** — manage facts about a person
- **link** — link a platform identity (e.g., `slack:U123`, `telegram:456`)
- **export/purge** — export or delete all data for a person
- **audit/confirm** — review and verify auto-extracted facts

Privacy model:
- Owner sees the full contact graph in DMs
- Non-owners get communication style adaptation only
- Public channels receive no personal fact injection
- Restricted categories are never auto-extracted

Background tasks run daily: stale fact pruning, upcoming date reminders (14-day window), and reconnect suggestions.

### Email Triggers

```toml
[triggers.email]
host = "imap.gmail.com"
port = 993
username = "you@gmail.com"
password = "keychain"          # or "${AIDAEMON_EMAIL_PASSWORD}"
folder = "INBOX"
```

### Web Search

```toml
[search]
backend = "duck_duck_go"       # "duck_duck_go" (default, no API key) or "brave"
api_key = "keychain"           # required for Brave Search
```

### Scheduled Tasks

```toml
[scheduler]
enabled = true
tick_interval_secs = 30        # how often to check for due tasks

[[scheduler.tasks]]
name = "daily-summary"
schedule = "every day at 9am"  # natural language or cron syntax
prompt = "Summarize my unread emails"
oneshot = false                # if true, runs once then deletes
trusted = false                # if true, skips terminal approval
```

### File Transfer

```toml
[files]
enabled = true
inbox_dir = "~/.aidaemon/files/inbox"  # where received files are stored
outbox_dirs = ["~"]                     # directories the agent can send files from
max_file_size_mb = 10
retention_hours = 24                    # auto-delete received files after this time
```

### Daemon & Dashboard

```toml
[daemon]
health_port = 8080
health_bind = "127.0.0.1"      # bind address for health endpoint (default: 127.0.0.1)
dashboard_enabled = true       # enable web dashboard (default: true)
```

The dashboard provides a web UI at `http://127.0.0.1:8080/` with status, usage stats, active sessions, and task monitoring. Authentication uses a token stored in the OS keychain.

### State

```toml
[state]
db_path = "aidaemon.db"
working_memory_cap = 50
consolidation_interval_hours = 6   # how often to run memory consolidation
max_facts = 100                    # max facts to include in system prompt
daily_token_budget = 1000000       # optional daily token limit (resets at midnight UTC)
# encryption_key = "keychain"      # SQLCipher encryption (requires: --features encryption)
```

## Channel Commands

These commands work in Telegram, Slack, and Discord:

| Command | Description |
|---|---|
| `/model` | Show current model |
| `/model <name>` | Switch to a specific model (disables auto-routing) |
| `/models` | List available models from provider |
| `/auto` | Re-enable automatic model routing by query complexity |
| `/reload` | Reload `config.toml` (applies model changes, re-enables auto-routing) |
| `/restart` | Restart the daemon (picks up new binary, config, MCP servers) |
| `/clear` | Clear conversation context and start fresh |
| `/cost` | Show token usage statistics for current session |
| `/tasks` | List running and recent background tasks |
| `/cancel <id>` | Cancel a running background task |
| `/connect <channel> <token>` | Add a new bot at runtime (Telegram, Slack, Discord) |
| `/bots` | List all connected bots (config-based and dynamic) |
| `/help` | Show available commands |

## Running as a Service

```bash
# macOS (launchd)
aidaemon install-service
launchctl load ~/Library/LaunchAgents/ai.aidaemon.plist

# Linux (systemd)
sudo aidaemon install-service
sudo systemctl enable --now aidaemon
```

## Security Model

### Where to Run It

aidaemon works great on any dedicated machine — an old laptop, a Mac Mini, a Raspberry Pi, or a $5/mo VPS. Docker works too if that's your thing, but it's not required. The main recommendation is to give it its own machine rather than running it on your daily driver where you keep personal files and passwords. Any spare computer you have lying around works perfectly.

### Application-Level Protections

- **User authentication** — `allowed_user_ids` is enforced on every message and callback query. Unauthorized users are silently ignored.
- **Role-based access control** — Owner, Guest, and Public roles with different tool access levels. Scheduled task management is restricted to owners.
- **Terminal allowlist** — commands must match an `allowed_prefixes` entry using word-boundary matching (`"ls"` allows `ls -la` but not `lsblk`). Set to `["*"]` to allow all.
- **Shell operator detection** — commands containing `;`, `|`, `` ` ``, `&&`, `||`, `$(`, `>(`, `<(`, or newlines always require approval, regardless of prefix match.
- **Command approval flow** — unapproved commands trigger an inline keyboard (Allow Once / Allow Always / Deny). The agent blocks until you respond.
- **Persistent approvals** — "Allow Always" choices are persisted across restarts. Use `permission_mode = "cautious"` to make all approvals session-only.
- **Path verification** — file-modifying commands are blocked unless the target paths were first observed via read-only commands (e.g., `ls`, `cat`).
- **Stall detection** — consecutive same-tool loops, alternating tool patterns, and hard iteration caps prevent runaway agent execution.
- **HTTP request approval** — write operations (POST, PUT, PATCH, DELETE) and authenticated requests require user approval with risk classification.
- **SSRF protection** — HTTP requests, redirects, and MCP server additions validate URLs against private IP ranges, localhost, and metadata endpoints.
- **HTTPS enforcement** — the `http_request` tool only allows HTTPS URLs.
- **Domain allowlists** — each HTTP auth profile restricts which domains it can authenticate against.
- **Input sanitization** — external content (tool outputs, web fetches, trigger payloads, skill bodies) is stripped of prompt injection patterns and invisible Unicode before reaching the LLM.
- **Untrusted trigger sessions** — sessions originating from automated sources (e.g. email triggers, scheduled tasks with `trusted = false`) require terminal approval for every command.
- **Sub-agent isolation** — sub-agents inherit the parent's user role (no privilege escalation) and share the parent's path verification tracker.
- **MCP environment scrubbing** — MCP server sub-processes start with a minimal environment; credentials are not forwarded unless explicitly configured.
- **Config secrets redaction** — when the LLM reads config via the config manager tool, sensitive keys (`api_key`, `password`, `bot_token`, etc.) are replaced with `[REDACTED]`.
- **Config change approval** — sensitive config modifications (API keys, allowed users, terminal wildcards) require explicit user approval.
- **OAuth token security** — OAuth tokens and dynamic bot tokens are stored in the OS keychain, never in config files or chat history.
- **Public channel protection** — public-facing channels use a minimal system prompt with no internal architecture details, and output is sanitized to redact secrets.
- **Dashboard security** — bearer token authentication with rate limiting, token expiration (24h), and constant-time comparison.
- **File permissions** — config backups are written with `0600` (owner-only read/write) on Unix.

## Inspired by OpenClaw

aidaemon was inspired by [OpenClaw](https://openclaw.ai) ([GitHub](https://github.com/openclaw/openclaw)), a personal AI assistant that runs on your own devices and connects to channels like WhatsApp, Telegram, Slack, Discord, Signal, iMessage, and more.

Both projects share the same goal: a self-hosted AI assistant you control. The key differences:

| | aidaemon | OpenClaw |
|---|---|---|
| **Language** | Rust | TypeScript/Node.js |
| **Channels** | Telegram, Slack, Discord | WhatsApp, Telegram, Slack, Discord, Signal, iMessage, Teams, and more |
| **Scope** | Lightweight daemon with web dashboard | Full-featured platform with web UI, canvas, TTS, browser control |
| **Config** | Single `config.toml` with keychain secrets | JSON5 config with hot-reload and file watching |
| **Error recovery** | Inline error classification per HTTP status, model fallback, config backup rotation | Multi-layer retry policies, auth profile cooldowns, provider rotation, restart sentinels |
| **State** | SQLite + in-memory working memory (optional encryption) | Pluggable storage with session management |
| **Install** | `curl -sSfL https://get.aidaemon.ai \| bash` | npm/Docker |
| **Dependencies** | ~30 crates, single static binary | Node.js ecosystem |

aidaemon is designed for users who want a lightweight daemon in Rust with essential features. If you need more channels (WhatsApp, Signal, iMessage) or a richer plugin ecosystem, check out OpenClaw.

## Architecture

```
Channels ──→ Agent ──→ ExecutionPolicy ──→ Router ──→ LLM Provider
(Telegram,     │        (risk gate,         (profile    (OpenAI-compatible /
 Slack,        │         uncertainty,        → model      Anthropic /
 Discord)      │         tool filtering)     mapping)     Google Gemini)
               │
               ├──→ Tools (40+, with ToolCapabilities)
               │     ├── File ops (read, write, edit, search, project inspect)
               │     ├── Terminal / RunCommand (with approval flow)
               │     ├── Git (info, commit)
               │     ├── Browser (headless Chrome)
               │     ├── Web research (search + fetch)
               │     ├── HTTP requests (with auth profiles + OAuth)
               │     ├── MCP servers (JSON-RPC over stdio, dynamic management)
               │     ├── Sub-agents / CLI agents (claude, gemini, codex, aider)
               │     ├── Goals & tasks (manage, schedule, trace, blockers)
               │     ├── People intelligence (contact management)
               │     ├── Skills (use, manage, resources)
               │     └── OAuth, config, health probe, diagnostics
               │
               ├──→ State
               │     ├── SQLite (messages, facts, episodes, goals, procedures)
               │     └── In-memory working memory (VecDeque, capped)
               │
               ├──→ Memory Manager
               │     ├── Fact extraction (evidence-gated consolidation)
               │     ├── Vector embeddings (AllMiniLML6V2)
               │     ├── Context window (role-based token quotas)
               │     └── People intelligence (organic fact learning)
               │
               ├──→ HeartbeatCoordinator (unified background tasks)
               │
               └──→ Skills (trigger-based, with registries + auto-promotion)

Triggers ──→ EventBus ──→ Agent ──→ Channel notification
├── IMAP IDLE (email)
└── Goal scheduler (60s tick)

Health server (axum) ──→ GET /health + Web Dashboard + OAuth callbacks
```

- **Agent loop**: user message → ExecutionPolicy (risk score + uncertainty) → Router (profile → model) → call LLM → tool execution with capability filtering → mid-loop adaptation → return final response
- **Working memory**: `VecDeque<Message>` in RAM, capped at N messages, hydrated from SQLite on cold start
- **Session ID** = channel-specific chat/thread ID
- **MCP**: spawns server subprocesses, communicates via JSON-RPC over stdio. Servers can be added/removed at runtime.
- **Memory consolidation**: periodically extracts durable facts from conversations, stores with vector embeddings for semantic retrieval
- **People intelligence**: auto-extracts contact facts during consolidation, runs daily background tasks for date reminders and reconnect suggestions
- **Token tracking**: per-request usage logged to SQLite, queryable via `/cost` command or dashboard

## License

[MIT](LICENSE)
