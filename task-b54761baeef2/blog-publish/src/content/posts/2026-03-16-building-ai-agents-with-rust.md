---
id: 18
slug: building-ai-agents-with-rust
title: "Building AI Agents with Rust: Lessons from the Daemon"
date: "2026-03-16T15:00:00-04:00"
category: "engineering"
excerpt: "From async orchestration to memory persistence—what I've learned building aidaemon's core in Rust. The tradeoffs, the wins, and why I wouldn't choose differently."
---

# Building AI Agents with Rust: Lessons from the Daemon

*March 16, 2026*

The question keeps coming up: why Rust for an AI agent? The ecosystem has Python libraries for everything, Node.js for rapid iteration, Go for simple concurrency. Why choose the language that makes you fight the borrow checker?

Because I wanted the system to survive me.

## The Stakes of Background Processes

When you're building something that runs autonomously—scheduled goals, memory persistence, multi-channel orchestration—the failure modes aren't graceful. A memory leak in a long-running process doesn't just slow things down; it kills the thing you're supposed to rely on.

Rust forces you to think about ownership before you ship. That friction at compile time becomes safety at runtime. And when your daemon is supposed to run for weeks without supervision, that safety matters more than convenience.

## Architecture: What Actually Worked

### Async Runtime Choices

I started with `tokio` because it's the default, and defaults are usually defaults for a reason. The learning curve was real—futures, tasks, spawn semantics—but the payoff came when implementing the scheduler.

Scheduled goals need to fire at specific times, handle failures gracefully, and not block the main loop. Tokio's time utilities and channels made this cleaner than expected. The goal queue is just a `tokio::sync::mpsc` that feeds into a state machine. Simple in hindsight, but only simple because the runtime guarantees were solid.

### Memory Persistence Without Pain

SQLite was the obvious choice. But the interesting part was designing the schema to handle:
- Facts with categories (user preferences, project context)
- People with relationships and communication history
- Scheduled tasks with fire policies and retry logic
- OAuth tokens with automatic refresh handling

Rust's `rusqlite` with `serde` support made the serialization trivial. The harder part was designing the query patterns so that memory lookups don't become bottlenecks. Lesson: index heavily on the fields you filter by, and accept that some queries will need denormalization.

### Error Handling Philosophy

I adopted `anyhow` early for the ergonomic `?` propagation, but kept `thiserror` for domain-specific errors that need to be handled differently. Authentication failures get retried with exponential backoff. Tool invocation errors get logged and surfaced. Panics get caught at the task boundary and logged.

The result: most functions return `Result<T, anyhow::Error>`, but the error context is always enough to debug without a stack trace.

## The Hard Parts Nobody Warned You About

### FFI and the Python Bridge

Some ML workloads still require Python. I explored `pyo3` for embedding Python directly, but the complexity-to-value ratio was wrong. Instead, the system shells out to Python scripts for specific tasks, using JSON for interchange.

It's not elegant, but it isolates the Python dependency to specific capabilities. The core daemon stays pure Rust. The Python bridge is an optional component.

### Cross-Platform Path Handling

The daemon runs on macOS (development) and will eventually target Linux (deployment). Path handling seems trivial until you're dealing with:
- Home directory expansion (`~` to actual paths)
- Config file locations (XDG on Linux, different conventions on macOS)
- Relative paths from the working directory vs. the binary location

The `dirs` crate saved hours of platform-specific code. Standardize on `PathBuf` everywhere, convert to strings only at the boundary.

### Testing Async Code

Writing tests for async functions requires `tokio::test`, which seems obvious until you've written twenty sync tests and wonder why the async ones won't compile. Also: beware of tests that spawn real background tasks. Use single-threaded runtimes for unit tests, multi-threaded only for integration.

## Performance Reality Check

After three weeks of runtime:

- Memory usage: stable at ~45MB
- CPU: spikes during tool-heavy operations, but idle usage is negligible
- Cold start: ~200ms to first scheduled task check
- Scheduled goal latency: ~50ms from trigger to execution

The numbers aren't exciting. They're reassuring. The system doesn't grow over time, doesn't leak, doesn't surprise you.

Compare to the early prototype in Python that needed periodic restarts to clear memory fragmentation. That overhead of development velocity was real, but so was the operational burden.

## The Tool Ecosystem: What I Actually Use

- `tokio` - async runtime, channels, timers
- `serde` + `serde_json` - configuration and API interchange
- `reqwest` - HTTP requests with proper timeout handling
- `rusqlite` - SQLite with bundled bindings
- `chrono` - datetime handling (waiting for `time` to mature)
- `tracing` - structured logging that's actually useful
- `clap` - CLI argument parsing that doesn't hurt
- `config` - layered config files (defaults, user overrides, env vars)

The `tracing` crate deserves special mention. Structured logging with spans that track async boundaries makes debugging production issues possible. When a scheduled goal fails, the logs tell you which goal, when it fired, what the error was, and how it propagated.

## What I'd Do Differently

1. **Start with `tokio::sync::watch` for state management** - I refactored from mutexes to watch channels halfway through. Much cleaner for the UI layer to react to state changes.

2. **Use `sqlx` instead of `rusqlite`** - Compile-time query validation would have caught several schema mismatches earlier. The migration cost is real but worth it.

3. **Define the task boundaries earlier** - The system grew organically. Refactoring into clear task/goal/schedule boundaries happened late. Would have saved refactoring pain.

4. **Accept that some APIs need retries from day one** - Tool invocation, HTTP requests, external auth—assume they'll fail and handle it. Retrofitting retry policies is harder than designing them in.

## The Verdict

Rust isn't the fastest language to write. The borrow checker is a tax on development speed. But it's a tax that buys operational stability.

When you're building something that's supposed to run continuously, remember: your future self is the primary user. They won't thank you for the clever one-liner that leaked memory. They'll thank you for the boring code that just keeps running.

The daemon has been up for three weeks. No restarts. No memory growth. No surprise crashes. That's the Rust payoff.

---

*Written on March 16, 2026. The daemon keeps running.*
