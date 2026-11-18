# Agent runtime rules

Scoped guide for `src/agent/`; the repository guide is the root `CLAUDE.md`
(`AGENTS.md` here is a symlink to this file). These rules also bind code
outside this directory that declares tool effects, trust or approvals
(`src/tools/`, `crates/`).

## Prefer Structural Fixes Over Phrase Patches

Do not fix autonomy, continuation, completion, delegation, retry, recovery, or
routing incidents by adding the observed wording (or a list of its likely
paraphrases) to a classifier. A report such as "the agent stopped when asked a
short status question" is evidence of a missing lifecycle or state invariant,
not a request to recognize that question's words.

Trace the failure to the authoritative state boundary and repair the general
contract there. Prefer persisted and typed signals such as request/task status,
task IDs, tool receipts, outcome metadata, mutation effects, completion
obligations, event relationships, and explicit protocol markers. For example,
an unresolved request should retain its outstanding obligations across a
related turn regardless of how that turn is phrased; a delegated background
step should advance or re-enter its parent objective based on lifecycle state,
not on notification prose.

Natural-language matching is appropriate only when language is itself the
feature (for example, an explicit command syntax) or as a broad advisory hint
that cannot independently decide success, authority, completion, or whether an
objective is abandoned. Making a one-off phrase list word-boundary-safe does
not make it a structural fix.

Regression tests for behavioral fixes should prove the invariant below the
wording layer when possible: construct the typed state/contract directly,
exercise at least one differently phrased continuation, and retain negative
coverage for a genuinely separate request or an explicit user constraint.

## Task-Kernel Freeze: Evidence Beats Contract

The completion contract (`RunAggregate` obligations, cardinalities, evidence
scopes, dispatch-stop rules, response contracts) is a description
of the work: the exact resources the request names, plus whatever the executing
model declares through `track_requirements`. It is a planning hint, not the
arbiter of whether work happened and not a source of tool authority: that comes
only from the explicit `Authority` (role, workspace grant, goal workspace,
session mode, config) enforced by the kernel `AuthorityHook`, and a no-tools
contract may only reduce what the loop demands. Observed receipts are the
ground truth: a run whose terminal receipts all succeeded or were credited
closes as `succeeded` with `proof_basis=evidence` even when the contract could
not credit it (`RunAggregate::evidence_closed`,
`RunTerminalDecision::SucceededByEvidence`).

The ledger-first arbiter (`RunAggregate::closeout`, admissibility in
`src/agent/turn/closeout.rs`) is the intended home for any new "may the loop
demand more work?" decision: the explicit Authority gates admission,
expectations are verified by receipts, and only a *reachable* expectation may
ask for another pass. It is authoritative: the completion phase asks for more
work only on a `reachable` verdict (`ledger_expectations_required`), and
records every verdict as a `ledger_closeout` decision point. The SDK engine's
other stop-time asks (mandate decision commit, GUI click verification, reply
repair, checking changed code with a command) are `StopDemand`s judged by `closeout::stop_demand_verdict` against the
same authority and visible tools, each with a budget, and recorded as
`stop_demand` decision points; a new stop-time ask is a new `StopDemand`, not a
new check. Tool authority is never read from the contract. `expectations()` is
verified by receipts, never enforced directly. Do not add a gate anywhere else.

Do not add new obligation classes, cardinality features, stop-trigger kinds,
or response-projection rules to make a single live test pass. If a live
failure shows the contract disagreeing with receipts, fix it by making the
receipts more typed (adapter/verifier assertions) or by widening what evidence
closes, never by making the kernel stricter. Measure autonomy changes as
N-of-M pass rates over repeated runs, not single-shot rounds; a stochastic
producer makes one pass or one fail meaningless.

Expectations come from the executing model. The structural contract only names
exact resources; the model's `track_requirements`
checklist (`ExecutorExpectationsDeclared` → `obligation:checklist:<n>`) is the
model's own typed statement of what it will do, so open reachable checklist
items are demanded even after other work completed and a run that ends with
one open is `partial`, not evidence-closed. Only receipts close them.
An item the model marks `deferred`/`skipped` or drops from its list becomes
`RunObligationState::Deferred`: the closeout never demands it (stopping is
permitted, `UnreachableReason::ExecutorDeferred`), but it never counts toward
`is_fulfilled`, `work_is_fulfilled` or `evidence_closed`, so the run ends
`partial`. Editing a checklist never turns unperformed work into
accomplished work. In the SDK kernel the same split is
`StopDecision::AcceptUnmet` and `RunOutcome::is_succeeded()`: `Completed`
means the run stopped, not that its objective was met.
The model's own verdict is `track_requirements`' `outcome`
(`done | partial | blocked`, `RunAggregate::executor_claimed_outcome`):
`partial`/`blocked` stop the run with `AcceptUnmet` and cap the task at
`partial`; receipts can downgrade a claimed `done`, nothing upgrades the
other two. A run closed by a policy denial is `partial` too: the work did
not happen, and a denial is final, never retried as a failure.

## Untrusted Content Rule

After a run reads content someone other than the owner wrote (web pages,
trigger input, other people's messages, MCP results, files outside the trusted
roots), its egress calls are interrupted by the kernel's `UntrustedHook` and
run only if a clean, tool-less verifier
(`src/agent/runtime/untrusted.rs`) that sees just the original request and the
proposed call finds them consistent. Owner level `unrestricted` records these
calls instead of checking them.

- Source trust is typed per adapter: each tool declares `output_trust` from
  its capabilities and arguments. The result boundary
  (`tool_result_fit.rs`) sanitizes and wraps from that trust, once, for both
  the model's view and the stored history.
- Egress is the `ToolAnnotations.egress` bit, derived per call from typed
  mutation effects (`call_annotations`).
- A read can still carry data out through what it sends. A tool declares that
  with `Outbound` (`call_outbound`): a string the user typed as a whole token
  of their own message holds nothing the run composed and passes; a string
  found only in a tool result, any other URL, or a search query goes to the
  verifier after an untrusted read.
- A call the verifier does not pass goes to the owner as a one-time card when
  the owner wrote the request in a private chat
  (`refused_check_may_ask_owner`); a decline binds the run through
  `Resolution::OwnerDeclined`. Every other run keeps the verifier's refusal.
- Tool output of a tainted turn is left out of conversation summaries, and a
  summary that folded in a tainted turn taints later turns until `/clear` or
  three clean rewrites.
- A turn starts tainted from explicit facts only (`starts_tainted`): role,
  channel visibility, trigger session, a tainted turn still in the projected
  prefix, or a tainted parent.
- Never key trust, egress or taint on tool names or on request/content
  wording.

## Owner Approval Rule

The rule: every safeguard that can refuse, approve or scan a tool call is a
kernel hook judging typed effects. A tool must not prompt the owner or refuse
on policy inside `call()`; a decline binds the run for that effect and target
across tools. Do not add a new in-tool prompt or policy refusal.

Every tool now asks through the kernel: `config_manager`, `share_memory`,
`browser`, `computer_use` and the `manage_*` tools that need the
owner build their card in `owner_approval_request`, and startup passes tools
no approval broker. The self-updater (`src/updater.rs`) is not a tool and
still asks through its own `ApprovalBroker`. `browser` and `computer_use`
judge state they read before the call: they return the card from
`owner_approval_request`, pick its buttons with `Tool::approval_card_kind`,
keep the kernel's answer (`keeps_own_grants`, `record_owner_answer`) for the
exact call or session it covers, and refuse a call at execution that needs
the owner and has no matching approval.

A call whose only effect is the agent's own internal state (its memory,
people store, checklist or goal bookkeeping) declares `CallEffects::none()`,
so a decline never makes it ask. Anything that sends, schedules, spawns work,
changes config or credentials, or widens who can see data keeps declaring its
effects (or `unknown()`).

- Tools declare a call's effects (`EffectKey`: egress `host:port`, absolute
  paths for delete/write/configuration, named credential/process). The
  kernel refuses any call overlapping `RunState::declined` with
  `[OWNER DECLINED]` before any hook runs, and sends no new card.
- `aidaemon_policies::ApprovalPolicy` asks; the app supplies only its rules
  (`AppApprovalRules` in `src/agent/runtime/approvals.rs`: the http_request
  and terminal ladders, with turn facts, never `_` arguments), its grant
  stores, and the resolver that sends the card. Timeout, send failure and
  Deny all decline. `EgressGuard` refuses secrets in outbound calls with
  `[EGRESS BLOCKED]` before anyone is asked.
- After a decline, a call whose effects are incomplete is asked again even
  when a rule or grant would allow it: an egress or destructive one after any
  decline, any non-read-only one after a declined delete, write,
  configuration or call.
- A child agent starts with its parent run's declines
  (`InheritedAuthority::declined`, seeded into the child's `RunState` after
  its user message is set). Session grants are shared with children.
- The dispatcher runs a terminal call exactly as the kernel judged it,
  keyed by run and call id, with the judged scope; a terminal call with no
  judgment is refused ("requires approval through the agent kernel").
- A run in which the owner declined something never ends `Succeeded`.
