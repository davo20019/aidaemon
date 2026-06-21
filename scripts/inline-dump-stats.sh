#!/usr/bin/env bash
# Count inline-dump-at-token-limit events and the fixable subset (those whose
# turn also spilled a large result). Measure-first input for the hard-cap decision.
#
# An "inline_dump" fires when the model hits the output token limit while answering
# inline (finish_reason=length, no tool calls). An "inline_dump_spill" fires when a
# large tool result was spilled to disk during that same turn. The "fixable" subset
# is turns that had BOTH: adding a hard cap on inline answers would have forced the
# model to use write_file instead, which is the desired behavior.
#
# Usage:  scripts/inline-dump-stats.sh [LOG_PATH]
#   LOG_PATH defaults to $AIDAEMON_LOG, then the macOS launchd location.
set -euo pipefail

LOG="${1:-${AIDAEMON_LOG:-$HOME/Library/Logs/aidaemon/stdout.log}}"
if [ ! -f "$LOG" ]; then
  echo "log not found: $LOG" >&2
  echo "pass the path as an argument or set AIDAEMON_LOG (e.g. ~/.local/state/aidaemon/stdout.log on Linux)" >&2
  exit 1
fi

# Strip ANSI once.
STRIPPED="$(sed 's/\x1b\[[0-9;]*m//g' "$LOG")"

# task_ids that hit an inline dump (exclude spill lines to avoid double-counting).
DUMP_TASKS="$(printf '%s\n' "$STRIPPED" | grep -F 'inline_dump' | grep -v 'inline_dump_spill' | grep -oE 'task_id=[0-9a-f-]+' | sort -u || true)"
SPILL_TASKS="$(printf '%s\n' "$STRIPPED" | grep -F 'inline_dump_spill' | grep -oE 'task_id=[0-9a-f-]+' | sort -u || true)"

TOTAL=$(printf '%s\n' "$DUMP_TASKS" | grep -c 'task_id=' || true)
FIXABLE=$(comm -12 <(printf '%s\n' "$DUMP_TASKS") <(printf '%s\n' "$SPILL_TASKS") | grep -c 'task_id=' || true)

echo "=== inline-dump telemetry ($LOG) ==="
echo "inline_dump events (unique turns):            $TOTAL"
echo "  of which had a spill same turn (fixable):   $FIXABLE"
echo
echo "=== output_tokens distribution (severity) ==="
TOKEN_DIST="$(printf '%s\n' "$STRIPPED" | grep -F 'inline_dump' | grep -v 'inline_dump_spill' | grep -oE 'output_tokens=[0-9]+' | sort -t= -k2 -n | uniq -c | tail -20 || true)"
if [ -n "$TOKEN_DIST" ]; then
  printf '%s\n' "$TOKEN_DIST"
else
  echo "  (no data)"
fi
echo
echo "=== by model ==="
MODEL_DIST="$(printf '%s\n' "$STRIPPED" | grep -F 'inline_dump' | grep -v 'inline_dump_spill' | grep -oE 'model=[^ ]+' | sort | uniq -c | sort -rn || true)"
if [ -n "$MODEL_DIST" ]; then
  printf '%s\n' "$MODEL_DIST"
else
  echo "  (no data)"
fi
echo
echo "=== by depth (0=root, >0=specialist) ==="
DEPTH_DIST="$(printf '%s\n' "$STRIPPED" | grep -F 'inline_dump' | grep -v 'inline_dump_spill' | grep -oE 'depth=[0-9]+' | sort | uniq -c || true)"
if [ -n "$DEPTH_DIST" ]; then
  printf '%s\n' "$DEPTH_DIST"
else
  echo "  (no data)"
fi
echo
echo "Decision rule:"
echo "  * TOTAL stays ~0 across real usage  -> inline token-limit hits are rare; no cap needed."
echo "  * TOTAL climbs, FIXABLE is high     -> hard cap justified; force write_file for long answers."
echo
echo "Inspect the actual dump cases:"
echo "  grep 'inline_dump ' \"$LOG\" | sed 's/\\x1b\\[[0-9;]*m//g'"
