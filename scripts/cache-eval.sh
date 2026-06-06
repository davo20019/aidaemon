#!/bin/sh
# cache-eval.sh — measure llama.cpp prompt-cache reuse for aidaemon agent-loop requests.
#
# Parses the llama-server log and prints, for each large (>MIN_TOKENS) request:
#   task=<id> prompt=<total tokens> reused_prefix=<n_past tokens> evaluated=<tokens re-evaluated> time_ms=<prompt eval ms>
#
# Healthy cache reuse: evaluated is small (hundreds) relative to prompt.
# Cache break: evaluated ~= prompt (full re-prefill, expect tens of seconds).
# reused_prefix stuck at the system-prompt size while evaluated stays large
# means the history region is being rewritten between iterations (e.g.
# sliding-window trimming or session-summary churn) — see
# docs/superpowers/specs/2026-06-06-cache-stable-system-prompt-design.md.
#
# Usage: scripts/cache-eval.sh [log-file] [min-prompt-tokens]
#   log-file          default: ~/.aidaemon/llama-server.log
#   min-prompt-tokens default: 5000 (filters out small auxiliary calls)

LOG="${1:-$HOME/.aidaemon/llama-server.log}"
MIN="${2:-5000}"

if [ ! -r "$LOG" ]; then
    echo "error: cannot read log file: $LOG" >&2
    exit 1
fi

awk -v min="$MIN" '
/new prompt/ {
    match($0, /task [0-9]+/); t = substr($0, RSTART + 5, RLENGTH - 5)
    match($0, /task.n_tokens = [0-9]+/); n = substr($0, RSTART + 16, RLENGTH - 16)
    npast = "-"
}
/n_past = [0-9]+, slot.prompt.tokens.size/ {
    match($0, /n_past = [0-9]+/); npast = substr($0, RSTART + 9, RLENGTH - 9)
}
/prompt eval time/ {
    match($0, /\/ +[0-9]+ tokens/); ev = substr($0, RSTART + 1, RLENGTH - 8); gsub(/[ \/]/, "", ev)
    match($0, /= +[0-9.]+ ms/); ms = substr($0, RSTART + 1, RLENGTH - 4); gsub(/[ =]/, "", ms)
    if (n + 0 > min) {
        printf "task=%s prompt=%s reused_prefix=%s evaluated=%s time_ms=%s\n", t, n, npast, ev, ms
        big++; total_ev += ev; total_ms += ms
        if (ev + 0 < n * 0.2) hits++
    }
}
END {
    if (big > 0) {
        printf "---\n%d large requests | %d cache hits (<20%% re-evaluated) | avg evaluated %.0f tokens | avg prompt phase %.1f s\n", \
            big, hits, total_ev / big, total_ms / big / 1000
    } else {
        print "no requests above threshold"
    }
}
' "$LOG"
