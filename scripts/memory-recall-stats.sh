#!/usr/bin/env bash
# Aggregate memory-recall telemetry to inform the Phase 2 (graph layer) decision.
#
# Reads the daemon's stdout log and tallies the `memory_recall`-target signals
# emitted by read-time neighborhood retrieval. The headline metric is
# "searched-but-denied": the model denied a named-person relational query even
# though a memory search returned results — the Phase-1 reasoning miss that an
# explicit graph (Phase 2) would address.
#
# Usage:  scripts/memory-recall-stats.sh [LOG_PATH]
#   LOG_PATH defaults to $AIDAEMON_LOG, then the macOS launchd location.
set -euo pipefail

LOG="${1:-${AIDAEMON_LOG:-$HOME/Library/Logs/aidaemon/stdout.log}}"
if [ ! -f "$LOG" ]; then
  echo "log not found: $LOG" >&2
  echo "pass the path as an argument or set AIDAEMON_LOG (e.g. ~/.local/state/aidaemon/stdout.log on Linux)" >&2
  exit 1
fi

data="$(sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep 'memory_recall:' || true)"

count() { echo "$data" | grep -c "$1" || true; }

searches=$(count 'explicit fact search reranked')
expansions=$(echo "$data" | grep 'neighborhood expansion' | grep -vc '(no entities)' || true)
no_entities=$(count 'neighborhood expansion (no entities)')
misses=$(count 'relational denial despite memory results')

echo "memory-recall telemetry  ($LOG)"
echo "  explicit searches (reranked):         $searches"
echo "  neighborhood expansions (added > 0):  $expansions"
echo "  expansions with no entities:          $no_entities"
echo "  SEARCHED-BUT-DENIED (Phase-1 miss):   $misses   <-- the Phase 2 signal"
echo
echo "Decision rule:"
echo "  * misses stays ~0 across real usage  -> Phase 1 suffices; do NOT build the graph."
echo "  * misses climbs (vs expansions)      -> Phase 1 is straining; Phase 2 graph is justified."
echo
echo "Inspect the actual miss cases (to confirm they are real reasoning failures,"
echo "not honest 'not in memory' denials):"
echo "  grep 'relational denial despite memory results' \"$LOG\" | sed 's/\\x1b\\[[0-9;]*m//g'"
