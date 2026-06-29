#!/usr/bin/env bash
# ensure-llama-flags.sh — idempotently ensure the local llama-server launchd job
# carries the flags aidaemon depends on. Safe to run repeatedly; a no-op when
# the flags are already present.
#
# Currently required: --kv-unified
#
# Why it matters: llama.cpp's idle-slot RAM cache (--cache-idle-slots, which
# parks a displaced slot's KV in the --cache-ram pool and restores it when its
# prefix returns) only activates with a unified KV buffer. --kv-unified defaults
# ON only when the slot count is "auto"; because aidaemon's launch sets
# --parallel explicitly, unified KV is OFF unless this flag is passed. Without
# it, a goal run's prompt prefix gets evicted by the interleaved background
# memory pipeline (cached≈0) and re-prefilled every call — inflating both
# latency and the per-goal token budget (observed: cached 9 -> ~11,800/call and
# goal budget 160k -> ~27k once enabled). Re-run this after recreating the
# llama plist by hand.
set -euo pipefail

PLIST="${LLAMA_PLIST:-$HOME/Library/LaunchAgents/ai.aidaemon.llama.plist}"
REQUIRED_FLAGS=("--kv-unified")

if [[ ! -f "$PLIST" ]]; then
  echo "llama plist not found at: $PLIST (nothing to do)"
  exit 0
fi

pb() { /usr/libexec/PlistBuddy -c "$1" "$PLIST"; }

changed=0
for flag in "${REQUIRED_FLAGS[@]}"; do
  if pb "Print :ProgramArguments" | grep -q -- "$flag"; then
    echo "present: $flag"
    continue
  fi
  if [[ $changed -eq 0 ]]; then
    cp "$PLIST" "$PLIST.bak.ensureflags.$(date +%s)"
  fi
  pb "Add :ProgramArguments: string $flag"
  echo "added:   $flag"
  changed=1
done

if [[ $changed -eq 1 ]]; then
  echo
  echo "Flags were added. Reload llama-server to apply (interrupts any run in flight):"
  echo "  launchctl bootout gui/\$(id -u)/ai.aidaemon.llama 2>/dev/null; \\"
  echo "  launchctl bootstrap gui/\$(id -u) \"$PLIST\""
else
  echo "All required llama-server flags already present."
fi
