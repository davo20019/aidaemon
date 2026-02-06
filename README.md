# aidaemon

[Website](https://aidaemon.ai/) · [Documentation](https://docs.aidaemon.ai/) · [GitHub](https://github.com/davo20019/aidaemon) · [𝕏](https://x.com/aidaemon_ai)

A personal AI agent that runs as a background daemon, accessible via Telegram, Slack, or Discord, with tool use, MCP integration, web research, scheduled tasks, and persistent memory.

I built this because I wanted to control my computer from my phone, from anywhere. I also wanted it to run on cheap hardware - a Raspberry Pi, an old laptop, a $5/month VPS - without eating all the RAM just to sit idle waiting for messages.

## Why Rust?

aidaemon runs 24/7 as a background daemon. It needs to be small, fast, and run on anything:

- **Runs on cheap/old hardware** - a lightweight Rust binary. On a Raspberry Pi or a $5 VPS with 512 MB RAM, it runs comfortably where heavier runtimes won't.
- **Single binary, zero runtime** - `cargo install aidaemon` gives you one binary. No Node.js, no Python, no Docker. Copy it to any machine and run it.
- **Startup in milliseconds** - restarts after a crash are near-instant, which matters for the auto-recovery retry loop.
- **No garbage collector** - predictable latency. No GC pauses between receiving the LLM response and sending the reply.

If you don't care about resource usage and want more channels (WhatsApp, Signal, iMessage) or a web canvas, check out [OpenClaw](https://openclaw.ai) which does similar things in TypeScript.

## Features

### Channels
- **Telegram interface** - chat with your AI assistant from any device
- **Slack integration** - Socket Mode support with threads, file sharing, and inline approvals (feature flag: `--features slack`)
- **Discord integration** - bot with slash commands and thread support (feature flag: `--features discord`)

### LLM Providers
- **Multiple providers** - native Anthropic, Google Gemini, DeepSeek, and OpenAI-compatible (OpenAI, OpenRouter, Ollama, etc.)
- **Smart model routing** - auto-selects Fast/Primary/Smart tier by query complexity (keyword heuristics, message length, code detection)
- **Token/cost tracking** - per-session and daily usage statistics with optional budget limits

### Tools & Agents
- **Agentic tool use** - the LLM can call tools (system info, terminal commands, MCP servers) in a loop
- **MCP client** - connect to any MCP server (filesystem, databases, etc.) and the agent gains those tools automatically
- **Browser tool** - headless Chrome with screenshot, click, fill, and JS execution
- **Web research** - search (DuckDuckGo/Brave) and fetch tools for internet access
- **Sub-agent spawning** - recursive agents with configurable depth, iteration limit, and dynamic budget extension
- **CLI agent delegation** - delegate tasks to claude, gemini, codex, aider, copilot (auto-discovered via `which`)
- **Skills system** - trigger-based markdown instructions loaded from a directory

### Memory & State
- **Persistent memory** - SQLite-backed conversation history + facts table, with fast in-memory working memory
- **Memory consolidation** - background fact extraction with vector embeddings (AllMiniLML6V2) for semantic recall
- **Database encryption** - optional SQLCipher AES-256 encryption at rest (feature flag: `--features encryption`)

### Automation
- **Scheduled tasks** - cron-style task scheduling with natural language time parsing
- **Email triggers** - IMAP IDLE monitors your inbox and notifies you on new emails
- **Background task registry** - track and cancel long-running tasks

### File Transfer
- **File sharing** - send and receive files through your chat channel
- **Configurable inbox/outbox** - control where files are stored and which directories the agent can access

### Security & Config
- **Config manager** - LLM can read/update `config.toml` with automatic backup, restore, and secrets redaction
- **Command approval flow** - inline keyboard (Allow Once / Allow Always / Deny) for unapproved terminal commands
- **Secrets management** - OS keychain integration + environment variable support for API keys

### Operations
- **Web dashboard** - built-in status page with usage stats, active sessions, and task monitoring
- **Channel commands** - `/model`, `/models`, `/auto`, `/reload`, `/restart`, `/clear`, `/cost`, `/tasks`, `/cancel`, `/help`
- **Auto-retry with backoff** - exponential backoff (5s → 10s → 20s → 40s → 60s cap) for dispatcher crashes
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
3. Setting up your Telegram bot token and user ID

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

Requires building with `--features slack`:

```toml
[slack]
enabled = true
app_token = "keychain"           # xapp-... Socket Mode token
bot_token = "keychain"           # xoxb-... Bot token
allowed_user_ids = ["U12345678"] # Slack user IDs (strings)
use_threads = true               # Reply in threads (default: true)
```

### Terminal Tool

```toml
[terminal]
# Set to ["*"] to allow all commands (only if you trust the LLM fully)
allowed_prefixes = ["ls", "cat", "head", "tail", "echo", "date", "whoami", "pwd", "find", "grep"]
```

### MCP Servers

```toml
[mcp.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
```

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

```toml
[skills]
enabled = true
dir = "skills"    # relative to config.toml location
```

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

- **User authentication** — `allowed_user_ids` is enforced on every message and callback query. Unauthorized users are silently ignored.
- **Terminal allowlist** — commands must match an `allowed_prefixes` entry using word-boundary matching (`"ls"` allows `ls -la` but not `lsblk`). Set to `["*"]` to allow all.
- **Shell operator detection** — commands containing `;`, `|`, `` ` ``, `&&`, `||`, `$(`, `>(`, `<(`, or newlines always require approval, regardless of prefix match.
- **Command approval flow** — unapproved commands trigger an inline keyboard (Allow Once / Allow Always / Deny). The agent blocks until you respond.
- **Persistent approvals** — "Allow Always" choices are persisted across restarts.
- **Untrusted trigger sessions** — sessions originating from automated sources (e.g. email triggers, scheduled tasks with `trusted = false`) require terminal approval for every command.
- **Config secrets redaction** — when the LLM reads config via the config manager tool, sensitive keys (`api_key`, `password`, `bot_token`, etc.) are replaced with `[REDACTED]`.
- **Config change approval** — sensitive config modifications (API keys, allowed users, terminal wildcards) require explicit user approval.
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
Channels ──→ Agent ──→ Smart Router ──→ LLM Provider
(Telegram,     │                         (OpenAI-compatible / Anthropic / Google Gemini)
 Slack,        │
 Discord)      ├──→ Tools
               │     ├── System info
               │     ├── Terminal (with approval flow)
               │     ├── Browser (headless Chrome)
               │     ├── Web research (search + fetch)
               │     ├── Config manager
               │     ├── MCP servers (JSON-RPC over stdio)
               │     ├── Sub-agents (recursive, depth-limited)
               │     ├── CLI agents (claude, gemini, codex, aider, copilot)
               │     └── Scheduler (create/list/cancel tasks)
               │
               ├──→ State
               │     ├── SQLite (conversation history + facts + usage)
               │     └── In-memory working memory (VecDeque, capped)
               │
               ├──→ Memory Manager
               │     ├── Fact extraction (background consolidation)
               │     └── Vector embeddings (AllMiniLML6V2)
               │
               ├──→ Task Registry (background task tracking)
               │
               └──→ Skills (trigger-based markdown instructions)

Triggers ──→ EventBus ──→ Agent ──→ Channel notification
├── IMAP IDLE (email)
└── Scheduler (cron tasks)

Health server (axum) ──→ GET /health + Web Dashboard
```

- **Agent loop**: user message → build history → smart router selects model tier → call LLM → if tool calls, execute and loop (max iterations) → return final response
- **Working memory**: `VecDeque<Message>` in RAM, capped at N messages, hydrated from SQLite on cold start
- **Session ID** = channel-specific chat/thread ID
- **MCP**: spawns server subprocesses, communicates via JSON-RPC over stdio
- **Memory consolidation**: periodically extracts durable facts from conversations, stores with vector embeddings for semantic retrieval
- **Token tracking**: per-request usage logged to SQLite, queryable via `/cost` command or dashboard

## License

[MIT](LICENSE)
