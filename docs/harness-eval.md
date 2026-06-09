# Harness evaluation framework

The harness eval framework measures **effectiveness** of the agent loop (routing, progress, contract fulfillment, cost) — not just LLM latency. Each completed task stores a `HarnessEvalSnapshot` on `TaskEnd`.

## Configuration

```toml
[diagnostics.harness_eval]
enabled = true
persist_on_task_end = true
weight_routing = 0.30
weight_progress = 0.25
weight_quality = 0.30
weight_cost = 0.15
cost_tier_cheap = 1.0
cost_tier_balanced = 2.5
cost_tier_strong = 5.0
cost_tier_unknown = 3.0
warn_overall_below = 0.6
warn_routing_below = 0.7
```

## Scores (0.0–1.0)

| Dimension | What it measures |
|-----------|------------------|
| `routing_accuracy` | Orchestration route vs actual tool use |
| `progress_yield` | Successful work per iteration |
| `contract_fulfillment` | Completion contract satisfied |
| `cost_efficiency` | Tier-weighted token waste |
| `overall` | Weighted composite (north-star metric) |

## Offline fixtures (CI regression)

Fixtures live in `tests/harness_eval/fixtures/*.yaml`.

```bash
cargo test --lib harness_eval
```

### Fixture shape

```yaml
name: my_case
session_id: eval_my_case_01
user_text: Check my system status
orchestrator: true          # use orchestrator-mode harness (routing tests)
mock_responses:
  - tool_call:
      name: system_info
      arguments: "{}"
  - text: Done.
expect:
  orchestration_route: default_continue
  tools_used: [system_info]
  outcome: succeeded
  overall_min: 0.5
```

**Assertion priority:** structural fields first (`orchestration_route`, `tools_used`, score mins), then optional `response_contains` (brittle — use sparingly).

## Online analysis

### db_probe

```bash
# Single task breakdown (requires encryption feature + AIDAEMON_ENCRYPTION_KEY)
cargo run --bin db_probe --features encryption -- --eval-task <task-id>

# Aggregate root-task scores (default: last 24h, root tasks only)
cargo run --bin db_probe --features encryption -- --eval-summary --eval-hours 168

# Include sub-agent tasks when debugging spawn flows
cargo run --bin db_probe --features encryption -- --eval-summary --eval-hours 24 --eval-include-subagents

# Record a fixture from a successful production run
cargo run --bin db_probe --features encryption -- \
  --record-fixture <session-id> [--task <task-id>] [--output tests/harness_eval/fixtures/my_case.yaml]
```

### diagnose tool

The `diagnose` action now appends a **Harness Effectiveness** section when `TaskEnd.harness_eval` is present, including score labels (ok/warn/bad) and routing mismatch warnings.

## Improvement workflow

1. **Baseline** — `cargo test --lib harness_eval` + `db_probe --eval-summary --eval-hours 168`
2. **Hypothesis** — e.g. "intent gate reduces partial outcomes on Change tasks"
3. **Change** — modify harness phase logic
4. **Verify** — fixtures pass; watch `routing_accuracy` / `contract_fulfillment` mins
5. **Ship** — add or update a fixture for every harness bug fix
6. **Monitor** — weekly `--eval-summary`; investigate if overall p50 drops >10%

### PR checklist

- [ ] `cargo test --lib harness_eval` passes
- [ ] No fixture `overall_min` regression without justification
- [ ] New harness behavior covered by a fixture when user-visible
- [ ] `CHANGELOG.md` updated when behavior changes

## Architecture notes

- Runtime accumulator: `src/agent/eval/accumulator.rs`
- Finalized snapshot attached in `emit_task_end` (`TaskEndData.harness_eval`)
- Sub-agent metrics roll up into the parent at spawn complete
- Root-task queries default to `depth = 0` / no `parent_task_id`
