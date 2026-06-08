#!/bin/sh
# cache-eval.sh — measure llama.cpp prompt-cache reuse for aidaemon agent-loop requests.
#
# Parses the llama-server log and prints, for each large (>MIN_TOKENS) request:
#   slot=<id> task=<id> prompt=<total tokens> reused_prefix=<cached tokens> evaluated=<tokens re-evaluated> time_ms=<prompt eval ms>
#
# Supports two llama-server log formats:
#   Legacy: "new prompt" + "n_past" + "prompt eval time" lines
#   Current (verbose): "slot print_timing ... prompt eval time" paired with
#     "stop processing: n_tokens" on the matching slot/task release line
#
# Healthy cache reuse: evaluated is small (hundreds) relative to prompt.
# Cache break: evaluated ~= prompt (full re-prefill, expect tens of seconds).
# reused_prefix stuck at the system-prompt size while evaluated stays large
# means the history region is being rewritten between iterations (e.g.
# sliding-window trimming or session-summary churn) — see
# docs/superpowers/specs/2026-06-06-cache-stable-system-prompt-design.md.
#
# Usage: scripts/cache-eval.sh [log-file] [min-prompt-tokens] [from-line]
#   log-file          default: ~/.aidaemon/llama-server.log
#   min-prompt-tokens default: 5000 (filters by total prompt/context size)
#   from-line         optional 1-based line number; analyze only from this line
#                       onward (handy after restarting llama-server with new flags)

LOG="${1:-$HOME/.aidaemon/llama-server.log}"
MIN="${2:-5000}"
FROM_LINE="${3:-1}"

if [ ! -r "$LOG" ]; then
    echo "error: cannot read log file: $LOG" >&2
    exit 1
fi

if [ "$FROM_LINE" -gt 1 ] 2>/dev/null; then
    AWK_INPUT=$(tail -n +"$FROM_LINE" "$LOG")
else
    AWK_INPUT=$(cat "$LOG")
fi

printf '%s\n' "$AWK_INPUT" | awk -v min="$MIN" '
function trim(s) {
    gsub(/^[ \t]+|[ \t]+$/, "", s)
    return s
}

# --- Legacy format (pre-verbose logging) ---
/new prompt/ {
    match($0, /task [0-9]+/)
    t = substr($0, RSTART + 5, RLENGTH - 5)
    match($0, /task\.n_tokens = [0-9]+/)
    n = substr($0, RSTART + 16, RLENGTH - 16)
    npast = "-"
    slot = "-"
    next
}
/n_past = [0-9]+, slot\.prompt\.tokens\.size/ {
    match($0, /n_past = [0-9]+/)
    npast = substr($0, RSTART + 9, RLENGTH - 9)
    next
}

# --- Current verbose format ---
/slot print_timing:.*prompt eval time/ {
    slot = task = ev = ms = ""
    if (match($0, /id[ \t]+[0-9]+/)) {
        slot = substr($0, RSTART, RLENGTH)
        gsub(/[^0-9]/, "", slot)
    }
    if (match($0, /task [0-9]+/)) {
        task = substr($0, RSTART + 5, RLENGTH - 5)
    }
    if (match($0, /\/[ \t]*[0-9]+ tokens/)) {
        ev = substr($0, RSTART + 1, RLENGTH - 1)
        gsub(/[^0-9]/, "", ev)
    }
    if (match($0, /=+[ \t]*[0-9.]+ ms/)) {
        ms = substr($0, RSTART, RLENGTH)
        gsub(/[^0-9.]/, "", ms)
    }
    if (slot != "" && task != "" && ev != "" && ms != "") {
        pending[slot SUBSEP task] = ev SUBSEP ms
    }
    next
}
/slot[ \t]+release:.*stop processing: n_tokens/ {
    slot = task = total = ""
    if (match($0, /id[ \t]+[0-9]+/)) {
        slot = substr($0, RSTART, RLENGTH)
        gsub(/[^0-9]/, "", slot)
    }
    if (match($0, /task [0-9]+/)) {
        task = substr($0, RSTART + 5, RLENGTH - 5)
    }
    if (match($0, /n_tokens = [0-9]+/)) {
        total = substr($0, RSTART + 11, RLENGTH - 11)
    }
    key = slot SUBSEP task
    if (total != "" && (key in pending)) {
        split(pending[key], parts, SUBSEP)
        ev = parts[1]
        ms = parts[2]
        npast = total - ev
        if (npast < 0) npast = 0
        if (total + 0 > min) {
            record(slot, task, total, npast, ev, ms)
        }
        delete pending[key]
    }
    next
}

# Legacy prompt eval (only when a preceding "new prompt" set n)
/prompt eval time/ {
    match($0, /\/[ \t]*[0-9]+ tokens/)
    ev = substr($0, RSTART + 1, RLENGTH - 1)
    gsub(/[^0-9]/, "", ev)
    match($0, /=+[ \t]*[0-9.]+ ms/)
    ms = substr($0, RSTART, RLENGTH)
    gsub(/[^0-9.]/, "", ms)
    if (n + 0 > min) {
        record(slot, t, n, npast, ev, ms)
    }
    next
}

function record(slot, task, prompt, reused, evaluated, time_ms,   reuse_pct) {
    reuse_pct = (prompt + 0 > 0) ? (100 * (prompt - evaluated) / prompt) : 0
    printf "slot=%s task=%s prompt=%s reused_prefix=%s evaluated=%s reuse_pct=%.1f time_ms=%s\n", \
        slot, task, prompt, reused, evaluated, reuse_pct, time_ms
    big++
    total_ev += evaluated + 0
    total_ms += time_ms + 0
    total_prompt += prompt + 0
    if (evaluated + 0 < prompt * 0.2) hits++
}

END {
    if (big > 0) {
        avg_reuse = (total_prompt > 0) ? (100 * (total_prompt - total_ev) / total_prompt) : 0
        printf "---\n%d large requests | %d cache hits (<20%% re-evaluated) | avg reuse %.1f%% | avg evaluated %.0f tokens | avg prompt phase %.1f s\n", \
            big, hits, avg_reuse, total_ev / big, total_ms / big / 1000
    } else {
        print "no requests above threshold"
    }
}
'
