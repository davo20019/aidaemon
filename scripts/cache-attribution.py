#!/usr/bin/env python3
"""cache-attribution.py — Phase 0 attribution: join aidaemon fingerprint logs
with llama.cpp prompt-cache stats and evaluate the sliding-window hysteresis
spec's exit criteria.

See docs/superpowers/specs/2026-06-06-sliding-window-hysteresis-design.md
("Phase 0 exit criterion"). This script implements item 5 (attribution run
analysis): every cache-break request on the llama-server side is matched, by
order, to the corresponding agent-loop LLM call on the daemon side and
classified by cause.

Inputs
------
1. Daemon log: aidaemon stdout captured to a file, run with
       RUST_LOG="info,aidaemon::agent::message_build_phase=debug"
   so the per-stage `Build stage pre-boundary fingerprint` debug lines are
   emitted alongside the info-level `Window decision` and
   `Provider-call prefix fingerprint` lines.
2. llama-server log (default ~/.aidaemon/llama-server.log), produced by a
   server running with `--parallel 1` (serialized, order-joinable).

Join validity (per spec): single active session, primary provider succeeds
without retries or cascade fallback, distinct user text per turn. The script
checks what it can (retry/cascade lines, join count mismatch) and reports
violations as run-invalidating.

Usage
-----
  scripts/cache-attribution.py --daemon-log run.log --session "telegram:123" \
      [--llama-log ~/.aidaemon/llama-server.log] [--llama-from-line N] \
      [--min-tokens 5000] [--break-threshold 0.2] [--json]
  scripts/cache-attribution.py --self-test

`--llama-from-line N` skips llama log lines before N (1-based); record
`wc -l < llama-server.log` immediately before the run and pass that +1.

Exit criteria evaluated
-----------------------
1. System-prompt stability: prefix_hash_system constant across within-task
   consecutive calls (same task, iteration >= 2). Cross-turn flips are
   reported separately and do not fail criterion 1.
2. Cause attribution: keep_from movement ALONE (no system churn, no tool-def
   refit, no identity-preserve drift change) accounts for >= 50% of
   attributable cache-break requests. Drift is reported separately.
3. Sample floor: >= 20 cache-break requests (prompt > min-tokens AND
   evaluated >= 20% of prompt).
"""

import argparse
import json
import re
import sys

# Build pipeline order of stage fingerprints (message_build_phase.rs).
# execution_checkpoint is a tail/full-payload hash, not pre-boundary.
# session_summary is intentionally excluded: Pillar A moves the session summary
# into the tail message (prefix_hash_archived / tail_hash split).  A
# session_summary-stage pre-boundary line would signal as
# content_mutation@session_summary, which is a *false* archived-region flip
# once the summary lives in the tail.  The tail_hash / prefix_hash_archived
# attribution path in attribute() handles that signal correctly instead.
STAGE_ORDER = [
    "age_collapse",
    "window_trim",
    "duplicate_removal",
    "json_conversion",
    "current_task_marker",
    "tool_error_collapse",
    "history_fitting",
]

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# key=value where value is "quoted", Some("quoted"), None, or a bare token.
FIELD_RE = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|Some\("(?:[^"\\]|\\.)*"\)|None|\S+)')

# Lines that invalidate an order-only join (multiple server requests for one
# LLM phase). Matched case-insensitively against WARN/ERROR daemon lines.
INVALIDATING_PATTERNS = [
    "cascade fallback",
    "retry",
    "retries exhausted",
    "falling back",
    "fallback model",
]


def parse_field_value(raw):
    if raw == "None":
        return None
    m = re.fullmatch(r'Some\("((?:[^"\\]|\\.)*)"\)', raw)
    if m:
        return m.group(1)
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    if raw in ("true", "false"):
        return raw == "true"
    try:
        return int(raw)
    except ValueError:
        return raw


def parse_fields(line):
    return {k: parse_field_value(v) for k, v in FIELD_RE.findall(line)}


def _new_pending():
    """Empty per-session build accumulator. `evicted`/`mutations`/
    `late_write_rerender` are the Pillar B expected-cause signals; they stay
    falsy/empty when the corresponding lines are absent, so old logs attribute
    exactly as before (back-compat)."""
    return {
        "stages": {},
        "window": None,
        "evicted": False,
        "mutations": [],
        "late_write_rerender": False,
    }


def parse_daemon_log(lines, session):
    """Returns (calls, invalid_lines, provider_calls).

    Each call is one provider-call fingerprint (filtered to `session`) with
    the window decision and stage hashes from the message build that preceded
    it, plus `call_idx`: the call's position in the daemon's FULL provider
    HTTP-call sequence (every `Calling LLM API` line to the primary URL —
    auxiliary intent/consultant/post-task calls included). Joining call_idx
    against the llama request sequence is exact even when aux calls are
    large, which an order-join over "large requests only" is not.
    """
    calls = []  # all sessions, for call_idx accounting
    invalid_lines = []
    pending = {}  # session -> {"stages": {...}, "window": {...}}
    pending_fp = None  # most recent fingerprint not yet matched to an HTTP call
    http_calls = []  # url of every `Calling LLM API` line, in order
    for lineno, raw in enumerate(lines, 1):
        line = ANSI_RE.sub("", raw.rstrip("\n"))
        if " WARN " in line or " ERROR " in line:
            low = line.lower()
            if any(p in low for p in INVALIDATING_PATTERNS):
                invalid_lines.append((lineno, line.strip()))
            continue
        if "Calling LLM API" in line:
            m = re.search(r"url=(\S+)", line)
            if pending_fp is not None:
                pending_fp["call_idx"] = len(http_calls)
                pending_fp = None
            http_calls.append(m.group(1) if m else None)
            continue
        if "Build stage pre-boundary fingerprint" in line:
            f = parse_fields(line)
            p = pending.setdefault(f.get("session_id"), _new_pending())
            p["stages"][f.get("stage")] = f.get("pre_boundary_hash")
        elif "Build stage tail fingerprint" in line:
            f = parse_fields(line)
            p = pending.setdefault(f.get("session_id"), _new_pending())
            p["stages"][f.get("stage")] = f.get("full_payload_hash")
        elif "Window decision" in line:
            f = parse_fields(line)
            p = pending.setdefault(f.get("session_id"), _new_pending())
            p["window"] = f
            # Pillar B: a `Window decision` carrying `turns_evicted > 0` is an
            # anchor advance — the oldest whole archived turns dropped, so the
            # prefix_hash_archived flip it causes is EXPECTED (eviction), not an
            # unattributed archived-region bug. The pre-Pillar-B `Window
            # decision` line (keep_from/oldest_kept_msg_id, no `turns_evicted`)
            # leaves this flag False, so old behaviour is unchanged.
            te = f.get("turns_evicted")
            if isinstance(te, int) and te > 0:
                p["evicted"] = True
        elif "Prefix mutation" in line:
            # Pillar B (Task 8): a retained stable-region mutator fired this
            # build (reason ∈ {repeated_tool_error_collapse, history_fitting,
            # empty_response_retry}). It rewrote stable-region bytes on purpose,
            # so the resulting archived flip is EXPECTED (logged mutation).
            f = parse_fields(line)
            p = pending.setdefault(f.get("session_id"), _new_pending())
            r = f.get("reason")
            if r is not None:
                p["mutations"].append(r)
        elif "archived render cache miss" in line:
            # Pillar B (Task 5/7): a render-cache miss with reason=fp_mismatch is
            # a late write / late completion record re-rendering exactly one
            # archived turn. The archived flip it causes is EXPECTED (logged
            # late-write re-render). Other miss reasons (miss / version_mismatch
            # / mode_mismatch) are cold-cache / version churn, not late writes,
            # and do not set the flag.
            f = parse_fields(line)
            p = pending.setdefault(f.get("session_id"), _new_pending())
            if f.get("reason") == "fp_mismatch":
                p["late_write_rerender"] = True
        elif "Provider-call prefix fingerprint" in line:
            f = parse_fields(line)
            sess = f.get("session_id")
            p = pending.pop(sess, _new_pending())
            call = {
                "lineno": lineno,
                "session": sess,
                "call_idx": None,
                "iteration": f.get("iteration"),
                "hash_system": f.get("prefix_hash_system"),
                "hash_pre_boundary": f.get("prefix_hash_pre_boundary"),
                # Pillar A fields: tail/archived split.  Empty string when
                # the tail message has not yet been introduced (old logs or
                # early tasks before the tail builder ships).
                "tail_hash": f.get("tail_hash") or "",
                "prefix_hash_archived": f.get("prefix_hash_archived") or "",
                "boundary_pos": f.get("boundary_pos"),
                "message_count": f.get("message_count"),
                "tool_defs_hash": f.get("tool_defs_hash"),
                "session_summary_hash": f.get("session_summary_hash"),
                "force_text": f.get("force_text"),
                "window": p["window"],
                "stages": p["stages"],
                # Pillar B expected-cause signals for the build that preceded
                # this provider call (all default to falsy/empty so absent lines
                # reproduce pre-Pillar-B attribution exactly — back-compat).
                "evicted": p["evicted"],
                "mutations": p["mutations"],
                "late_write_rerender": p["late_write_rerender"],
            }
            calls.append(call)
            pending_fp = call
    # Primary URL = the URL of the first fingerprint-attached call (the
    # llama-server). Calls to other URLs (cascade fallbacks) are dropped from
    # the sequence, and call_idx is remapped to primary-URL-only positions.
    primary_url = next(
        (
            http_calls[c["call_idx"]]
            for c in calls
            if c["call_idx"] is not None
        ),
        None,
    )
    primary_positions = {}  # original index -> primary-only index
    n = 0
    for i, url in enumerate(http_calls):
        if url == primary_url:
            primary_positions[i] = n
            n += 1
    for c in calls:
        c["call_idx"] = primary_positions.get(c["call_idx"])
    target = [c for c in calls if c["session"] == session]
    return target, invalid_lines, n


def assign_tasks(calls):
    """Segment calls into tasks. Prefer the window decision's current_turn_id;
    fall back to the iteration-reset heuristic (iteration <= previous)."""
    task = 0
    prev = None
    for c in calls:
        if prev is not None:
            t_prev = (prev.get("window") or {}).get("current_turn_id")
            t_cur = (c.get("window") or {}).get("current_turn_id")
            if t_prev is not None and t_cur is not None:
                if t_cur != t_prev:
                    task += 1
            elif isinstance(c["iteration"], int) and isinstance(prev["iteration"], int):
                if c["iteration"] <= prev["iteration"]:
                    task += 1
            else:
                task += 1
        c["task"] = task
        prev = c
    return calls


def parse_llama_log(lines, min_tokens, from_line=1):
    """One record per completed request, keyed by llama task id.

    Two log verbosities are supported:
    - Verbose builds emit `new prompt ... task.n_tokens = N` (total prompt)
      and `n_past = N, slot.prompt.tokens.size` (reused prefix) directly —
      same source lines as scripts/cache-eval.sh.
    - Current INFO-level builds omit both; we derive them from lines that are
      always present: prompt = release n_tokens - generated tokens, and
      reused = prompt - evaluated. Records are emitted at slot release.
    """
    reqs = []
    state = {}  # task id -> partial record
    last_task = None  # --parallel 1 serializes requests; older builds split
    # print_timing across physical lines whose continuations carry no task id.

    def task_id(line):
        m = re.search(r"task (\d+)", line)
        return m.group(1) if m else None

    for lineno, line in enumerate(lines, 1):
        if lineno < from_line:
            continue
        t_line = task_id(line)
        if t_line is not None:
            last_task = t_line
        if "new prompt" in line:
            t = t_line or last_task
            m = re.search(r"task\.n_tokens = (\d+)", line)
            state[t] = {"total": int(m.group(1)) if m else None}
        elif re.search(r"n_past = \d+, slot\.prompt\.tokens\.size", line):
            t = t_line or last_task
            m = re.search(r"n_past = (\d+)", line)
            state.setdefault(t, {})["npast"] = int(m.group(1)) if m else None
        elif "prompt eval time" in line:
            t = t_line or last_task
            m = re.search(r"/\s*(\d+) tokens", line)
            ev = int(m.group(1)) if m else 0
            m = re.search(r"=\s*([0-9.]+) ms", line)
            ms = float(m.group(1)) if m else 0.0
            st = state.setdefault(t, {})
            st["evaluated"] = ev
            st["ms"] = ms
        elif "eval time" in line and "total time" not in line:
            # generation-phase print_timing line ("eval time = ... / M tokens")
            t = t_line or last_task
            m = re.search(r"/\s*(\d+) tokens", line)
            if m and t in state:
                state[t]["generated"] = int(m.group(1))
        elif "stop processing: n_tokens =" in line:
            t = t_line or last_task
            st = state.pop(t, None)
            if st is None or "evaluated" not in st:
                # Released without a prompt-eval (e.g. cancelled mid-prefill).
                # Emit a placeholder so 1:1 request ordering is preserved.
                reqs.append(
                    {
                        "task": t,
                        "prompt": None,
                        "reused_prefix": None,
                        "evaluated": None,
                        "time_ms": None,
                    }
                )
                continue
            total = st.get("total")
            if total is None:
                m = re.search(r"n_tokens = (\d+)", line)
                release_n = int(m.group(1)) if m else 0
                total = release_n - st.get("generated", 0)
            reused = st.get("npast")
            if reused is None:
                reused = total - st["evaluated"]
            if total > min_tokens:
                reqs.append(
                    {
                        "task": t,
                        "prompt": total,
                        "reused_prefix": reused,
                        "evaluated": st["evaluated"],
                        "time_ms": st["ms"],
                    }
                )
    return reqs


def attribute(call, prev):
    """Classify what changed between consecutive provider calls. Returns
    (primary_cause, changes, keep_from_alone).

    Pillar A tail/archived split logic
    -----------------------------------
    When both calls carry non-empty prefix_hash_archived values, we can
    distinguish tail-only flips from true archived-region changes:

    * prefix_hash_archived stable AND tail_hash changed
      → tail_replacement (expected): the volatile tail message rotated; this
        is the normal and expected per-turn behaviour.  It never counts toward
        pre_boundary_changed_unattributed.
    * prefix_hash_archived changed
      → archived region changed; falls through to the normal cause ladder
        (system / tool_defs / keep_from / drift / stage / unattributed).

    When either call has an empty prefix_hash_archived (old log, no tail
    builder yet), the field is absent and attribution falls back to the
    pre-Pillar-A hash_pre_boundary path — behaviour unchanged for old logs.
    """
    changes = []
    system = call["hash_system"] != prev["hash_system"]
    tool_defs = call["tool_defs_hash"] != prev["tool_defs_hash"]
    if system:
        changes.append("system")
    if tool_defs:
        changes.append("tool_defs")

    w, pw = call.get("window"), prev.get("window")
    keep_from_moved = False
    drift = False
    if w is not None and pw is not None:
        # oldest_kept_msg_id is the identity signal: keep_from is an index
        # and can hold the same number while the fetch window slides.
        if w.get("oldest_kept_msg_id") != pw.get("oldest_kept_msg_id") or w.get(
            "keep_from"
        ) != pw.get("keep_from"):
            keep_from_moved = True
            changes.append("keep_from")
        if w.get("identity_preserve_bypass") != pw.get("identity_preserve_bypass"):
            drift = True
            changes.append("identity_drift")

    divergent_stage = None
    for s in STAGE_ORDER:
        a, b = call["stages"].get(s), prev["stages"].get(s)
        if a is not None and b is not None and a != b:
            divergent_stage = s
            break
    if divergent_stage:
        changes.append(f"stage:{divergent_stage}")
    if call["session_summary_hash"] != prev["session_summary_hash"]:
        changes.append("session_summary")

    pre_boundary_stable = call["hash_pre_boundary"] == prev["hash_pre_boundary"]

    # Pillar A: tail/archived split.  Only applies when both calls carry
    # non-empty prefix_hash_archived (i.e. after the tail builder ships).
    archived_cur = call.get("prefix_hash_archived") or ""
    archived_prev = prev.get("prefix_hash_archived") or ""
    tail_cur = call.get("tail_hash") or ""
    tail_prev = prev.get("tail_hash") or ""
    both_have_archived = bool(archived_cur) and bool(archived_prev)

    if both_have_archived:
        archived_stable = archived_cur == archived_prev
        tail_changed = tail_cur != tail_prev
        both_tail_nonempty = bool(tail_cur) and bool(tail_prev)
        if archived_stable and tail_changed and both_tail_nonempty:
            # Tail-only flip: volatile per-turn context rotated as expected.
            # Requires both tail values non-empty — a disappearing or newly
            # appearing tail (one side empty) could mask a builder regression
            # and must fall through to the normal cause ladder instead.
            # This must NOT fall through to pre_boundary_changed_unattributed.
            changes.append("tail_only")
            return "tail_replacement (expected)", changes, False

    # Pillar B: an archived-region flip is EXPECTED iff this build logged one of
    # the three accounted-for causes (spec exit criterion: every
    # prefix_hash_archived flip pairs with a logged eviction, a Prefix mutation,
    # or a render-cache fp_mismatch; a flip with NONE of these is an
    # archived-region bug → pre_boundary_changed_unattributed). This classifies
    # ONLY the archived flip that would otherwise be unattributed — a real
    # higher-priority cause (system / tool_defs / keep_from / drift / stage) in
    # the same build still wins below. When no archived flip occurred, or the
    # signals are absent (old logs), this resolves to None and behaviour is
    # unchanged.
    archived_changed = both_have_archived and archived_cur != archived_prev
    expected_archived_cause = None
    expected_archived_change = None
    if archived_changed:
        if call.get("evicted"):
            expected_archived_cause = "eviction (expected)"
            expected_archived_change = "eviction"
        elif call.get("mutations"):
            expected_archived_cause = "prefix_mutation (expected)"
            expected_archived_change = "prefix_mutation:" + ",".join(
                call["mutations"]
            )
        elif call.get("late_write_rerender"):
            expected_archived_cause = "late_write_rerender (expected)"
            expected_archived_change = "late_write_rerender"

    if system:
        primary = "system_prompt_churn"
    elif tool_defs:
        primary = "tool_defs_refit"
    elif keep_from_moved:
        primary = "keep_from_movement" + ("+drift" if drift else "")
    elif drift:
        primary = "identity_preserve_drift"
    elif divergent_stage:
        primary = f"content_mutation@{divergent_stage}"
    elif pre_boundary_stable:
        primary = "post_build_or_tail" + ("(force_text)" if call["force_text"] else "")
    elif expected_archived_cause is not None:
        # Archived region flipped, but a logged eviction / Prefix mutation /
        # fp_mismatch accounts for it — expected, NOT an unattributed bug.
        primary = expected_archived_cause
        changes.append(expected_archived_change)
    else:
        primary = "pre_boundary_changed_unattributed"

    keep_from_alone = keep_from_moved and not (system or tool_defs or drift)
    return primary, changes, keep_from_alone


def analyze(calls, reqs, min_tokens, break_threshold, provider_calls=None):
    """`reqs` is the FULL llama request sequence (no size filter); each call
    joins by its call_idx position in that sequence."""

    def req_is_break(r):
        return (
            r is not None
            and r["prompt"] is not None
            and r["prompt"] > min_tokens
            and r["evaluated"] >= break_threshold * r["prompt"]
        )

    calls = assign_tasks(calls)
    joined = []
    for i, c in enumerate(calls):
        r = (
            reqs[c["call_idx"]]
            if c["call_idx"] is not None and c["call_idx"] < len(reqs)
            else None
        )
        entry = {
            "idx": i,
            "call_idx": c["call_idx"],
            "task": c["task"],
            "iteration": c["iteration"],
            "prompt": r["prompt"] if r else None,
            "reused_prefix": r["reused_prefix"] if r else None,
            "evaluated": r["evaluated"] if r else None,
            "time_ms": r["time_ms"] if r else None,
            "is_break": req_is_break(r),
            "force_text": c["force_text"],
            "primary_cause": None,
            "changes": [],
            "keep_from_alone": False,
            "within_task": False,
        }
        if i > 0:
            prev = calls[i - 1]
            entry["within_task"] = (
                c["task"] == prev["task"]
                and isinstance(c["iteration"], int)
                and c["iteration"] >= 2
            )
            primary, changes, kfa = attribute(c, prev)
            entry["primary_cause"] = primary
            entry["changes"] = changes
            entry["keep_from_alone"] = kfa
        joined.append(entry)

    # Criterion 3 counts cache-break requests as observed on the llama side,
    # independent of join completeness.
    breaks = [r for r in reqs if req_is_break(r)]
    joined_breaks = [e for e in joined if e["is_break"]]
    attributable_breaks = [e for e in joined_breaks if e["primary_cause"] is not None]
    kf_alone_breaks = [e for e in attributable_breaks if e["keep_from_alone"]]
    drift_breaks = [
        e for e in attributable_breaks if "identity_drift" in e["changes"]
    ]
    # Tail-only flips (Pillar A): expected rotations, counted separately.
    tail_only_flips = [
        e for e in joined if e["primary_cause"] == "tail_replacement (expected)"
    ]

    # Criterion 1: prefix_hash_system flips on within-task consecutive pairs.
    within_flips = []
    cross_flips = []
    for i in range(1, len(calls)):
        if calls[i]["hash_system"] == calls[i - 1]["hash_system"]:
            continue
        if (
            calls[i]["task"] == calls[i - 1]["task"]
            and isinstance(calls[i]["iteration"], int)
            and calls[i]["iteration"] >= 2
        ):
            within_flips.append(i)
        else:
            cross_flips.append(i)

    pct = (
        100.0 * len(kf_alone_breaks) / len(attributable_breaks)
        if attributable_breaks
        else 0.0
    )
    n_large = sum(
        1 for r in reqs if r["prompt"] is not None and r["prompt"] > min_tokens
    )
    return {
        "joined": joined,
        "n_calls": len(calls),
        "n_requests": len(reqs),
        "n_large_requests": n_large,
        "n_provider_calls": provider_calls,
        "join_mismatch": provider_calls is not None
        and provider_calls != len(reqs),
        "n_breaks": len(breaks),
        "n_joined_breaks": len(joined_breaks),
        "n_attributable_breaks": len(attributable_breaks),
        "n_keep_from_alone": len(kf_alone_breaks),
        "n_drift_breaks": len(drift_breaks),
        "n_tail_only_flips": len(tail_only_flips),
        "keep_from_pct": pct,
        "within_task_system_flips": within_flips,
        "cross_turn_system_flips": cross_flips,
        "criterion1_pass": len(within_flips) == 0,
        "criterion2_pass": bool(attributable_breaks) and pct >= 50.0,
        "criterion3_pass": len(breaks) >= 20,
    }


def report(res, invalid_lines):
    join_mismatch = res["join_mismatch"]
    out = []
    out.append("=== Phase 0 attribution report ===")
    out.append(
        f"daemon agent-loop calls: {res['n_calls']} | "
        f"daemon provider HTTP calls: {res['n_provider_calls']} | "
        f"llama requests: {res['n_requests']} "
        f"({res['n_large_requests']} large)"
    )
    if join_mismatch:
        out.append(
            "!! JOIN COUNT MISMATCH — daemon HTTP-call count != llama "
            "request count. Check for concurrent daemons, dropped/errored "
            "requests, or a stale --llama-from-line offset; re-run under "
            "controlled conditions."
        )
    if invalid_lines:
        out.append(
            f"!! RUN INVALID — {len(invalid_lines)} retry/cascade line(s) "
            "in daemon log (multiple server requests per LLM phase):"
        )
        for ln, text in invalid_lines[:10]:
            out.append(f"     line {ln}: {text[:160]}")
    out.append("")
    out.append("per-request join (break requests marked *):")
    out.append(
        f"{'idx':>4} {'task':>4} {'iter':>4} {'prompt':>7} {'reused':>7} "
        f"{'eval':>7} {'brk':>3}  cause / changes"
    )
    for e in res["joined"]:
        cause = e["primary_cause"] or "(first call — no prior)"
        ch = ",".join(e["changes"])

        def col(v):
            return str(v) if v is not None else "-"

        out.append(
            f"{e['idx']:>4} {e['task']:>4} {col(e['iteration']):>4} "
            f"{col(e['prompt']):>7} {col(e['reused_prefix']):>7} "
            f"{col(e['evaluated']):>7} "
            f"{'*' if e['is_break'] else '':>3}  {cause}"
            + (f" [{ch}]" if ch else "")
        )
    out.append("")
    causes = {}
    for e in res["joined"]:
        if e["is_break"] and e["primary_cause"]:
            causes[e["primary_cause"]] = causes.get(e["primary_cause"], 0) + 1
    out.append("cache-break cause breakdown:")
    for cause, n in sorted(causes.items(), key=lambda kv: -kv[1]):
        out.append(f"  {n:>3}  {cause}")
    if not causes:
        out.append("  (no attributable cache breaks)")
    out.append(
        f"tail-only flips (expected): {res['n_tail_only_flips']}"
        + (
            "  [Pillar A: volatile tail rotations — not cache-break signals]"
            if res["n_tail_only_flips"] > 0
            else ""
        )
    )
    out.append("")
    c1, c2, c3 = (
        res["criterion1_pass"],
        res["criterion2_pass"],
        res["criterion3_pass"],
    )
    out.append("=== Exit criteria ===")
    out.append(
        f"1. system-prompt stability (within-task): "
        f"{'PASS' if c1 else 'FAIL'} "
        f"({len(res['within_task_system_flips'])} within-task flip(s); "
        f"{len(res['cross_turn_system_flips'])} cross-turn flip(s), "
        f"reported separately, not a criterion-1 failure)"
    )
    out.append(
        f"2. keep_from movement alone >= 50% of breaks: "
        f"{'PASS' if c2 else 'FAIL'} "
        f"({res['n_keep_from_alone']}/{res['n_attributable_breaks']} "
        f"attributable breaks = {res['keep_from_pct']:.0f}%; "
        f"{res['n_drift_breaks']} drift break(s) measured separately)"
    )
    out.append(
        f"3. sample floor >= 20 cache-break requests: "
        f"{'PASS' if c3 else 'FAIL'} ({res['n_breaks']} observed)"
    )
    verdict = (
        c1 and c2 and c3 and not invalid_lines and not join_mismatch
    )
    out.append("")
    out.append(
        "VERDICT: "
        + (
            "Phase 0 exit criteria MET — Phase 1 anchor work is authorized."
            if verdict
            else "Phase 0 exit criteria NOT met — do not ship the anchor on "
            "spec faith; design for the dominant measured cause."
        )
    )
    return "\n".join(out)


# ----------------------------------------------------------------------------
# Self-test fixtures
# ----------------------------------------------------------------------------


def _stage_lines(sess, it, hashes):
    lines = []
    for s in STAGE_ORDER:
        lines.append(
            f"2026-06-06T20:00:00.000000Z DEBUG aidaemon::agent::message_build_phase: "
            f'Build stage pre-boundary fingerprint session_id="{sess}" '
            f'iteration={it} stage="{s}" pre_boundary_hash={hashes.get(s, "h0")}'
        )
    lines.append(
        f"2026-06-06T20:00:00.000000Z DEBUG aidaemon::agent::message_build_phase: "
        f'Build stage tail fingerprint session_id="{sess}" iteration={it} '
        f'stage="execution_checkpoint" full_payload_hash=tail{it}'
    )
    return lines


def _window_line(sess, it, turn, keep_from, oldest_kept, bypass=0):
    return (
        f"2026-06-06T20:00:00.000000Z  INFO aidaemon::agent::message_build_phase: "
        f'Window decision session_id="{sess}" iteration={it} '
        f'current_turn_id=Some("{turn}") boundary_msg_id=Some("b1") '
        f'oldest_fetched_id=Some("f1") oldest_kept_msg_id=Some("{oldest_kept}") '
        f"keep_from={keep_from} window_size=4 identity_preserve_bypass={bypass} "
        f"history_limit=40 fetched_count=30 current_user_injected=false "
        f"safe_collapse=false"
    )


def _eviction_window_line(sess, it, turns_evicted=1, new_anchor=4):
    """Pillar B `Window decision` line emitted when the anchor advances.

    Mirrors the info! at message_build_phase.rs (fields: new_anchor,
    turns_evicted, kept_est_tokens, archived_kept, archived_budget). This is the
    eviction-flavoured `Window decision`, distinct from the legacy keep_from
    variant in `_window_line`.
    """
    return (
        f"2026-06-06T20:00:00.000000Z  INFO aidaemon::agent::message_build_phase: "
        f'Window decision session_id="{sess}" iteration={it} '
        f"new_anchor={new_anchor} turns_evicted={turns_evicted} "
        f"kept_est_tokens=3000 archived_kept=2 archived_budget=4000"
    )


def _prefix_mutation_line(sess, it, reason):
    """Pillar B (Task 8) `Prefix mutation` line — a retained stable-region
    mutator fired (reason ∈ {repeated_tool_error_collapse, history_fitting,
    empty_response_retry}). Mirrors the info! at message_build_phase.rs."""
    return (
        f"2026-06-06T20:00:00.000000Z  INFO aidaemon::agent::message_build_phase: "
        f'Prefix mutation session_id="{sess}" iteration={it} reason="{reason}"'
    )


def _render_cache_miss_line(sess, turn_id="turn-1", reason="fp_mismatch"):
    """Pillar B (Task 5/7) `archived render cache miss` line. reason=fp_mismatch
    is the late-write re-render signal. Mirrors the info! at
    message_build_phase.rs (fields: turn_id, reason)."""
    return (
        f"2026-06-06T20:00:00.000000Z  INFO aidaemon::agent::message_build_phase: "
        f'archived render cache miss session_id="{sess}" '
        f'turn_id=Some("{turn_id}") reason="{reason}"'
    )


def _provider_line(
    sess,
    it,
    hsys,
    hpre,
    tdefs="td1",
    ssum="ss1",
    force="false",
    tail_hash="",
    prefix_hash_archived="",
):
    """Emit a Provider-call prefix fingerprint log line.

    tail_hash / prefix_hash_archived are the Pillar A fields added in Task 1.
    Omitting them (or passing empty strings) replicates old-log behaviour so
    backwards-compat tests can verify the parser handles absent fields cleanly.
    """
    tail_part = f" tail_hash={tail_hash}" if tail_hash else " tail_hash="
    archived_part = (
        f" prefix_hash_archived={prefix_hash_archived}"
        if prefix_hash_archived
        else " prefix_hash_archived="
    )
    return (
        f"2026-06-06T20:00:00.000000Z  INFO aidaemon::agent::llm_phase: "
        f'Provider-call prefix fingerprint session_id="{sess}" iteration={it} '
        f"prefix_hash_system={hsys} prefix_hash_pre_boundary={hpre}"
        f"{archived_part}{tail_part}"
        f" boundary_pos=10 message_count=20 tool_defs_hash={tdefs} "
        f"session_summary_hash={ssum} force_text={force}"
    )


def _calling_line(url="http://127.0.0.1:8081/v1/chat/completions", tools=20):
    return (
        f"2026-06-06T20:00:00.000000Z  INFO aidaemon::providers::openai_compatible: "
        f'Calling LLM API model="gemma-4-26b" url={url} tools={tools} '
        f"response_mode=Text tool_choice=Auto"
    )


def _llama_req(task, prompt, npast, evaluated, verbose=True, generated=50):
    lines = []
    if verbose:
        lines.append(
            f"slot update_slots: id  0 | task {task} | new prompt, n_ctx_slot = 65536, "
            f"n_keep = 0, task.n_tokens = {prompt}"
        )
        lines.append(
            f"slot update_slots: id  0 | task {task} | n_past = {npast}, "
            f"slot.prompt.tokens.size() = {prompt}, seq_id = 0, pos_min = 0, n_swa = 1024"
        )
    lines.append(
        f"slot print_timing: id  0 | task {task} | prompt eval time =    1000.00 ms "
        f"/  {evaluated} tokens (    1.00 ms per token,  1000.00 tokens per second)"
    )
    lines.append(
        f"slot print_timing: id  0 | task {task} |        eval time =     500.00 ms "
        f"/     {generated} tokens (   10.00 ms per token,   100.00 tokens per second)"
    )
    lines.append(
        f"slot print_timing: id  0 | task {task} |       total time =    1500.00 ms "
        f"/   {evaluated + generated} tokens"
    )
    lines.append(
        f"slot      release: id  0 | task {task} | stop processing: "
        f"n_tokens = {prompt + generated}, truncated = 0"
    )
    return lines


def self_test():
    sess = "telegram:1"
    daemon = []
    llama = []
    # Task A (turn t1), 3 iterations.
    # it1: baseline (first call — unattributable).
    daemon += _stage_lines(sess, 1, {})
    daemon.append(_window_line(sess, 1, "t1", 0, "m1"))
    daemon.append(_provider_line(sess, 1, "sysA", "preA"))
    daemon.append(_calling_line())
    llama += _llama_req(1, 10000, 0, 10000)  # break, first call
    # Auxiliary LLM call (intent gate etc.): no fingerprint, still hits llama.
    # The call_idx join must skip over it without misaligning.
    daemon.append(_calling_line(tools=0))
    llama += _llama_req(2, 900, 0, 900, verbose=False)  # small aux request
    # it2: keep_from moved alone -> break attributed keep_from_movement.
    daemon += _stage_lines(sess, 2, {"window_trim": "h_changed"})
    daemon.append(_window_line(sess, 2, "t1", 2, "m3"))
    daemon.append(_provider_line(sess, 2, "sysA", "preB"))
    daemon.append(_calling_line())
    llama += _llama_req(3, 11000, 800, 9000)  # break
    # it3: everything stable -> tail-only growth, cache hit.
    daemon += _stage_lines(sess, 3, {"window_trim": "h_changed"})
    daemon.append(_window_line(sess, 3, "t1", 2, "m3"))
    daemon.append(_provider_line(sess, 3, "sysA", "preB"))
    daemon.append(_calling_line())
    llama += _llama_req(4, 12000, 11000, 1000)  # hit
    # Task B (turn t2): cross-turn system flip (not a criterion-1 failure)
    daemon += _stage_lines(sess, 1, {"window_trim": "h_changed"})
    daemon.append(_window_line(sess, 1, "t2", 2, "m3"))
    daemon.append(_provider_line(sess, 1, "sysB", "preC"))
    daemon.append(_calling_line())
    llama += _llama_req(5, 13000, 500, 12500)  # break (system churn)
    # it2 of task B: drift-only change -> drift break.
    daemon += _stage_lines(sess, 2, {"window_trim": "h_changed"})
    daemon.append(_window_line(sess, 2, "t2", 2, "m3", bypass=2))
    daemon.append(_provider_line(sess, 2, "sysB", "preD"))
    daemon.append(_calling_line())
    llama += _llama_req(6, 13500, 500, 13000)  # break (drift)
    # it3 of task B: keep_from moved alone again -> break.
    daemon += _stage_lines(sess, 3, {"window_trim": "h_zz"})
    daemon.append(_window_line(sess, 3, "t2", 4, "m5", bypass=2))
    daemon.append(_provider_line(sess, 3, "sysB", "preE"))
    daemon.append(_calling_line())
    llama += _llama_req(7, 14000, 500, 13500)  # break

    calls, invalid, provider_calls = parse_daemon_log(daemon, sess)
    assert len(calls) == 6, f"expected 6 calls, got {len(calls)}"
    assert provider_calls == 7, f"expected 7 provider calls, got {provider_calls}"
    assert [c["call_idx"] for c in calls] == [0, 2, 3, 4, 5, 6]
    assert not invalid
    assert calls[0]["stages"]["age_collapse"] == "h0"
    assert calls[1]["stages"]["window_trim"] == "h_changed"
    assert calls[0]["window"]["current_turn_id"] == "t1"
    assert calls[0]["window"]["keep_from"] == 0
    assert calls[0]["stages"]["execution_checkpoint"] == "tail1"

    reqs = parse_llama_log(llama, 0)
    assert len(reqs) == 7, f"expected 7 requests, got {len(reqs)}"
    assert reqs[3]["reused_prefix"] == 11000
    assert reqs[1]["prompt"] == 900  # aux request present in full sequence

    # INFO-level fallback: no "new prompt"/n_past lines; prompt and reused
    # derived from print_timing + release lines.
    fb = parse_llama_log(
        _llama_req(9, 8000, None, 1500, verbose=False, generated=100), 5000
    )
    assert len(fb) == 1
    assert fb[0]["prompt"] == 8000  # release n_tokens (8100) - generated (100)
    assert fb[0]["reused_prefix"] == 6500  # 8000 - 1500 evaluated
    # Small auxiliary calls stay filtered in fallback mode too.
    assert not parse_llama_log(
        _llama_req(10, 900, None, 900, verbose=False, generated=10), 5000
    )

    res = analyze(calls, reqs, 5000, 0.2, provider_calls)
    assert not res["join_mismatch"]
    j = res["joined"]
    assert j[0]["is_break"] and j[0]["primary_cause"] is None
    assert j[1]["primary_cause"] == "keep_from_movement" and j[1]["keep_from_alone"]
    assert not j[2]["is_break"]
    assert j[3]["primary_cause"] == "system_prompt_churn"
    assert j[4]["primary_cause"] == "identity_preserve_drift"
    assert j[5]["primary_cause"] == "keep_from_movement" and j[5]["keep_from_alone"]
    # Tasks: calls 0-2 task 0, calls 3-5 task 1.
    assert [c["task"] for c in calls] == [0, 0, 0, 1, 1, 1]
    # Criterion 1: no within-task flips; one cross-turn flip.
    assert res["criterion1_pass"]
    assert len(res["cross_turn_system_flips"]) == 1
    # Criterion 2: attributable breaks = idx 1,3,4,5 -> keep_from alone = 2/4 = 50%.
    assert res["n_attributable_breaks"] == 4
    assert res["n_keep_from_alone"] == 2
    assert res["criterion2_pass"]
    # Criterion 3: only 5 breaks < 20.
    assert res["n_breaks"] == 5
    assert not res["criterion3_pass"]

    # Retry/cascade detection invalidates the run.
    bad = daemon + [
        "2026-06-06T20:01:00.000000Z  WARN aidaemon::agent::llm: "
        "Transient error retries exhausted, trying cascade fallback"
    ]
    _, invalid2, _ = parse_daemon_log(bad, sess)
    assert len(invalid2) == 1

    # Fallback-provider calls (different URL) are excluded from the join
    # sequence: provider_calls unchanged, the cascaded call gets no call_idx.
    casc = daemon + [
        _provider_line(sess, 4, "sysB", "preF"),
        _calling_line(url="https://openrouter.ai/api/v1/chat/completions"),
    ]
    calls3, _, pc3 = parse_daemon_log(casc, sess)
    assert pc3 == 7
    assert calls3[-1]["call_idx"] is None

    # Quoted-stage and None field parsing edge cases.
    f = parse_fields(
        'x session_id="a b" v=None t=Some("q r") n=3 b=true h=abc123'
    )
    assert f == {
        "session_id": "a b",
        "v": None,
        "t": "q r",
        "n": 3,
        "b": True,
        "h": "abc123",
    }

    # ------------------------------------------------------------------
    # Pillar A: tail_hash / prefix_hash_archived attribution tests.
    # ------------------------------------------------------------------

    # Test A1: tail-only flip → tail_replacement (expected).
    # Archived region is stable; only the tail message changed.
    c_tail_prev = {
        "hash_system": "sysX",
        "hash_pre_boundary": "preX_full",
        "prefix_hash_archived": "archX",
        "tail_hash": "tailX_v1",
        "tool_defs_hash": "td1",
        "session_summary_hash": "ss1",
        "force_text": False,
        "window": None,
        "stages": {},
    }
    c_tail_cur = {
        "hash_system": "sysX",
        "hash_pre_boundary": "preX_full_v2",  # pre_boundary changed (tail included)
        "prefix_hash_archived": "archX",       # archived region STABLE
        "tail_hash": "tailX_v2",               # tail changed
        "tool_defs_hash": "td1",
        "session_summary_hash": "ss1",
        "force_text": False,
        "window": None,
        "stages": {},
    }
    cause_a1, changes_a1, kfa_a1 = attribute(c_tail_cur, c_tail_prev)
    assert cause_a1 == "tail_replacement (expected)", (
        f"tail-only flip: expected 'tail_replacement (expected)', got {cause_a1!r}"
    )
    assert "tail_only" in changes_a1
    assert not kfa_a1, "tail_replacement must not set keep_from_alone"

    # Test A2: archived region changed → falls through to normal cause ladder
    # (pre_boundary_changed_unattributed when no other signal).
    c_arch_prev = {
        "hash_system": "sysX",
        "hash_pre_boundary": "preY_v1",
        "prefix_hash_archived": "archY_v1",
        "tail_hash": "tailY",
        "tool_defs_hash": "td1",
        "session_summary_hash": "ss1",
        "force_text": False,
        "window": None,
        "stages": {},
    }
    c_arch_cur = {
        "hash_system": "sysX",
        "hash_pre_boundary": "preY_v2",
        "prefix_hash_archived": "archY_v2",   # archived region CHANGED
        "tail_hash": "tailY",                  # tail stable
        "tool_defs_hash": "td1",
        "session_summary_hash": "ss1",
        "force_text": False,
        "window": None,
        "stages": {},
    }
    cause_a2, _, _ = attribute(c_arch_cur, c_arch_prev)
    assert cause_a2 == "pre_boundary_changed_unattributed", (
        f"archived flip: expected 'pre_boundary_changed_unattributed', got {cause_a2!r}"
    )

    # Test A3: backwards compatibility — absent fields (old logs, empty strings).
    # No prefix_hash_archived → falls back to hash_pre_boundary path.
    c_old_prev = {
        "hash_system": "sysZ",
        "hash_pre_boundary": "preZ_v1",
        "prefix_hash_archived": "",   # absent / old log
        "tail_hash": "",
        "tool_defs_hash": "td1",
        "session_summary_hash": "ss1",
        "force_text": False,
        "window": None,
        "stages": {},
    }
    c_old_cur = {
        "hash_system": "sysZ",
        "hash_pre_boundary": "preZ_v2",   # changed (would look like tail flip)
        "prefix_hash_archived": "",       # absent — tail logic must NOT fire
        "tail_hash": "",
        "tool_defs_hash": "td1",
        "session_summary_hash": "ss1",
        "force_text": False,
        "window": None,
        "stages": {},
    }
    cause_a3, _, _ = attribute(c_old_cur, c_old_prev)
    assert cause_a3 == "pre_boundary_changed_unattributed", (
        f"old-log compat: expected 'pre_boundary_changed_unattributed', got {cause_a3!r}"
    )

    # Test A3b: archived stable + tail_prev non-empty + tail_cur empty
    # (tail disappeared) — must NOT classify as tail_replacement (expected).
    # A disappearing tail could signal a builder regression; fall through to the
    # normal cause ladder (pre_boundary_changed_unattributed here).
    c_tail_disappear_prev = {
        "hash_system": "sysW",
        "hash_pre_boundary": "preW_v1",
        "prefix_hash_archived": "archW",
        "tail_hash": "tailW_v1",     # non-empty tail before
        "tool_defs_hash": "td1",
        "session_summary_hash": "ss1",
        "force_text": False,
        "window": None,
        "stages": {},
    }
    c_tail_disappear_cur = {
        "hash_system": "sysW",
        "hash_pre_boundary": "preW_v2",
        "prefix_hash_archived": "archW",   # archived stable
        "tail_hash": "",                   # tail disappeared
        "tool_defs_hash": "td1",
        "session_summary_hash": "ss1",
        "force_text": False,
        "window": None,
        "stages": {},
    }
    cause_a3b, _, _ = attribute(c_tail_disappear_cur, c_tail_disappear_prev)
    assert cause_a3b != "tail_replacement (expected)", (
        f"tail disappearance: must NOT be 'tail_replacement (expected)', got {cause_a3b!r}"
    )

    # Test A4: tail-only flip does NOT count toward pre_boundary_changed_unattributed
    # in a full analyze() run.  Build a minimal 3-call sequence:
    #   call 0: baseline (no prior)
    #   call 1: tail-only flip  → tail_replacement (expected)
    #   call 2: archived flip   → pre_boundary_changed_unattributed
    sess2 = "telegram:99"
    daemon_pa = []
    llama_pa = []
    daemon_pa += _stage_lines(sess2, 1, {})
    daemon_pa.append(_window_line(sess2, 1, "tPA", 0, "mPA1"))
    daemon_pa.append(
        _provider_line(
            sess2, 1, "sysPAa", "prePAa_full",
            tail_hash="tailPA_v1", prefix_hash_archived="archPA",
        )
    )
    daemon_pa.append(_calling_line())
    llama_pa += _llama_req(101, 10000, 0, 10000)
    # call 1: tail-only flip (archived stable, tail changed, pre_boundary changed).
    daemon_pa += _stage_lines(sess2, 2, {})
    daemon_pa.append(_window_line(sess2, 2, "tPA", 0, "mPA1"))
    daemon_pa.append(
        _provider_line(
            sess2, 2, "sysPAa", "prePAa_full_v2",
            tail_hash="tailPA_v2", prefix_hash_archived="archPA",
        )
    )
    daemon_pa.append(_calling_line())
    llama_pa += _llama_req(102, 11000, 0, 11000)  # break (tail-only)
    # call 2: archived region changed.
    daemon_pa += _stage_lines(sess2, 3, {})
    daemon_pa.append(_window_line(sess2, 3, "tPA", 0, "mPA1"))
    daemon_pa.append(
        _provider_line(
            sess2, 3, "sysPAa", "prePAa_full_v3",
            tail_hash="tailPA_v2", prefix_hash_archived="archPA_v2",
        )
    )
    daemon_pa.append(_calling_line())
    llama_pa += _llama_req(103, 12000, 0, 12000)  # break (archived flip)

    calls_pa, _, pc_pa = parse_daemon_log(daemon_pa, sess2)
    assert len(calls_pa) == 3, f"PA: expected 3 calls, got {len(calls_pa)}"
    assert calls_pa[0]["tail_hash"] == "tailPA_v1"
    assert calls_pa[0]["prefix_hash_archived"] == "archPA"
    assert calls_pa[1]["tail_hash"] == "tailPA_v2"
    assert calls_pa[1]["prefix_hash_archived"] == "archPA"
    assert calls_pa[2]["prefix_hash_archived"] == "archPA_v2"

    reqs_pa = parse_llama_log(llama_pa, 0)
    res_pa = analyze(calls_pa, reqs_pa, 5000, 0.2, pc_pa)
    j_pa = res_pa["joined"]
    assert j_pa[1]["primary_cause"] == "tail_replacement (expected)", (
        f"PA call 1: expected tail_replacement, got {j_pa[1]['primary_cause']!r}"
    )
    assert j_pa[2]["primary_cause"] == "pre_boundary_changed_unattributed", (
        f"PA call 2: expected pre_boundary_changed_unattributed, got {j_pa[2]['primary_cause']!r}"
    )
    assert res_pa["n_tail_only_flips"] == 1, (
        f"expected 1 tail-only flip, got {res_pa['n_tail_only_flips']}"
    )
    assert not j_pa[1]["keep_from_alone"], "tail_replacement must not set keep_from_alone"
    # tail-only breaks inflate the attributable denominator (by design) but not the numerator
    assert res_pa["n_attributable_breaks"] == 2  # call 1 (tail) + call 2 (archived)
    assert res_pa["keep_from_pct"] == 0.0  # no keep_from_alone breaks in this run

    # Verify report includes the tail-only flips line.
    report_text = report(res_pa, [])
    assert "tail-only flips (expected):" in report_text, (
        "report must include 'tail-only flips (expected):' summary line"
    )
    count_str = report_text.split("tail-only flips (expected):")[1].strip().split()[0]
    assert count_str == "1", (
        f"tail-only flips count must be '1' in report, got {count_str!r}"
    )

    # ------------------------------------------------------------------
    # Pillar B (Task 9): eviction / prefix_mutation / late_write_rerender
    # expected causes for archived-region flips.
    # ------------------------------------------------------------------

    # Test B1: archived flip + eviction Window decision (turns_evicted>0)
    # → eviction (expected), NOT pre_boundary_changed_unattributed.
    # Under Pillar B the legacy keep_from `Window decision` is gone: a window
    # line is emitted ONLY when the anchor advances (eviction). Non-eviction
    # builds emit no window line, so `window` stays None and the keep_from/drift
    # comparison never fires — the archived flip is classified purely by the
    # eviction / mutation / fp_mismatch signals (or unattributed if absent).
    sessB = "telegram:7"
    daemon_b = []
    llama_b = []
    daemon_b += _stage_lines(sessB, 1, {})
    daemon_b.append(
        _provider_line(
            sessB, 1, "sysB1", "preB1",
            tail_hash="tailB", prefix_hash_archived="archB1",
        )
    )
    daemon_b.append(_calling_line())
    llama_b += _llama_req(201, 10000, 0, 10000)
    # call 1: anchor advances (eviction) → archived flips, expected.
    daemon_b += _stage_lines(sessB, 2, {})
    daemon_b.append(_eviction_window_line(sessB, 2, turns_evicted=2, new_anchor=5))
    daemon_b.append(
        _provider_line(
            sessB, 2, "sysB1", "preB2",
            tail_hash="tailB", prefix_hash_archived="archB2",
        )
    )
    daemon_b.append(_calling_line())
    llama_b += _llama_req(202, 9000, 0, 9000)  # break (eviction)
    # call 2: archived flips with a Prefix mutation logged → expected.
    daemon_b += _stage_lines(sessB, 3, {})
    daemon_b.append(_prefix_mutation_line(sessB, 3, "history_fitting"))
    daemon_b.append(
        _provider_line(
            sessB, 3, "sysB1", "preB3",
            tail_hash="tailB", prefix_hash_archived="archB3",
        )
    )
    daemon_b.append(_calling_line())
    llama_b += _llama_req(203, 9000, 0, 9000)  # break (prefix_mutation)
    # call 3: archived flips with a render-cache fp_mismatch logged → expected.
    daemon_b += _stage_lines(sessB, 4, {})
    daemon_b.append(_render_cache_miss_line(sessB, "turn-old", "fp_mismatch"))
    daemon_b.append(
        _provider_line(
            sessB, 4, "sysB1", "preB4",
            tail_hash="tailB", prefix_hash_archived="archB4",
        )
    )
    daemon_b.append(_calling_line())
    llama_b += _llama_req(204, 9000, 0, 9000)  # break (late_write_rerender)
    # call 4: archived flips with NO accounting signal → unattributed bug.
    daemon_b += _stage_lines(sessB, 5, {})
    daemon_b.append(
        _provider_line(
            sessB, 5, "sysB1", "preB5",
            tail_hash="tailB", prefix_hash_archived="archB5",
        )
    )
    daemon_b.append(_calling_line())
    llama_b += _llama_req(205, 9000, 0, 9000)  # break (unattributed)

    calls_b, _, pc_b = parse_daemon_log(daemon_b, sessB)
    assert len(calls_b) == 5, f"B: expected 5 calls, got {len(calls_b)}"
    # Per-build signals captured on the right call.
    assert calls_b[1]["evicted"] is True, "eviction Window decision not captured"
    assert calls_b[0]["evicted"] is False
    assert calls_b[2]["mutations"] == ["history_fitting"]
    assert calls_b[3]["late_write_rerender"] is True
    assert calls_b[4]["evicted"] is False
    assert calls_b[4]["mutations"] == []
    assert calls_b[4]["late_write_rerender"] is False

    reqs_b = parse_llama_log(llama_b, 0)
    res_b = analyze(calls_b, reqs_b, 5000, 0.2, pc_b)
    jb = res_b["joined"]
    assert jb[1]["primary_cause"] == "eviction (expected)", (
        f"B call 1: expected eviction (expected), got {jb[1]['primary_cause']!r}"
    )
    assert "eviction" in jb[1]["changes"]
    assert jb[2]["primary_cause"] == "prefix_mutation (expected)", (
        f"B call 2: expected prefix_mutation (expected), got {jb[2]['primary_cause']!r}"
    )
    assert any(c.startswith("prefix_mutation:") for c in jb[2]["changes"])
    assert jb[3]["primary_cause"] == "late_write_rerender (expected)", (
        f"B call 3: expected late_write_rerender (expected), got {jb[3]['primary_cause']!r}"
    )
    assert "late_write_rerender" in jb[3]["changes"]
    assert jb[4]["primary_cause"] == "pre_boundary_changed_unattributed", (
        f"B call 4: archived flip with no signal must stay unattributed, "
        f"got {jb[4]['primary_cause']!r}"
    )

    # Test B2: a real higher-priority cause in the SAME build as an eviction
    # still wins (the expected-cause path only classifies the otherwise-
    # unattributed flip — it must not mask system/tool_defs/keep_from churn).
    c_evict_sys_prev = {
        "hash_system": "sysQ_v1",
        "hash_pre_boundary": "preQ_v1",
        "prefix_hash_archived": "archQ_v1",
        "tail_hash": "tailQ",
        "tool_defs_hash": "td1",
        "session_summary_hash": "ss1",
        "force_text": False,
        "window": None,
        "stages": {},
        "evicted": False,
        "mutations": [],
        "late_write_rerender": False,
    }
    c_evict_sys_cur = {
        "hash_system": "sysQ_v2",          # system churned too
        "hash_pre_boundary": "preQ_v2",
        "prefix_hash_archived": "archQ_v2",  # archived flipped
        "tail_hash": "tailQ",
        "tool_defs_hash": "td1",
        "session_summary_hash": "ss1",
        "force_text": False,
        "window": None,
        "stages": {},
        "evicted": True,                    # eviction logged
        "mutations": [],
        "late_write_rerender": False,
    }
    cause_b2, _, _ = attribute(c_evict_sys_cur, c_evict_sys_prev)
    assert cause_b2 == "system_prompt_churn", (
        f"B2: real system churn must win over eviction, got {cause_b2!r}"
    )

    # Test B3: back-compat — old logs (no eviction/mutation/fp_mismatch lines)
    # with an archived flip still classify as pre_boundary_changed_unattributed.
    # This re-uses the Pillar A archived-flip dicts (no Pillar B keys present),
    # so attribute() reads the signals via .get() → None and behaves as before.
    cause_b3, _, _ = attribute(c_arch_cur, c_arch_prev)
    assert cause_b3 == "pre_boundary_changed_unattributed", (
        f"B3 back-compat: archived flip with no signals must be unattributed, "
        f"got {cause_b3!r}"
    )

    # Test B4: back-compat — a legacy `Window decision` (keep_from variant, no
    # `turns_evicted`) must NOT set the eviction flag.
    legacy_only = _stage_lines("telegram:8", 1, {}) + [
        _window_line("telegram:8", 1, "tL", 0, "mL1"),
        _provider_line("telegram:8", 1, "sysL", "preL"),
        _calling_line(),
    ]
    calls_legacy, _, _ = parse_daemon_log(legacy_only, "telegram:8")
    assert calls_legacy[0]["evicted"] is False, (
        "legacy keep_from Window decision must not set the eviction flag"
    )

    print(report(res, []))
    print("\nself-test: PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--daemon-log", help="aidaemon stdout capture")
    ap.add_argument("--session", help="session_id to filter daemon logs by")
    ap.add_argument(
        "--llama-log",
        default=None,
        help="llama-server log (default ~/.aidaemon/llama-server.log)",
    )
    ap.add_argument(
        "--llama-from-line",
        type=int,
        default=1,
        help="skip llama log lines before this 1-based line number",
    )
    ap.add_argument("--min-tokens", type=int, default=5000)
    ap.add_argument("--break-threshold", type=float, default=0.2)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.daemon_log or not args.session:
        ap.error("--daemon-log and --session are required (or use --self-test)")

    import os

    llama_log = args.llama_log or os.path.expanduser("~/.aidaemon/llama-server.log")
    with open(args.daemon_log, errors="replace") as f:
        calls, invalid_lines, provider_calls = parse_daemon_log(f, args.session)
    with open(llama_log, errors="replace") as f:
        # Parse the FULL request sequence (no size filter) — the size filter
        # applies at analysis time; the join needs every request.
        reqs = parse_llama_log(f, 0, args.llama_from_line)

    res = analyze(calls, reqs, args.min_tokens, args.break_threshold, provider_calls)
    if args.json:
        res["invalid_lines"] = invalid_lines
        print(json.dumps(res, indent=2))
    else:
        print(report(res, invalid_lines))


if __name__ == "__main__":
    main()
