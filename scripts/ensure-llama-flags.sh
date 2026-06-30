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
# Bare flags (no value).
#  --kv-unified: enables the idle-slot RAM cache (see below).
#  --mlock: pin the model in RAM so macOS can't compress/swap it under memory
#    pressure — avoids latency spikes (steadier tail latency). Non-fatal if it
#    can't lock (llama.cpp warns and continues).
REQUIRED_FLAGS=("--kv-unified" "--mlock")
# Key/value flags: "flag value".
#  --reasoning-budget 256: caps chain-of-thought to ~256 tokens (~8s decode vs
#    ~30s); thinking is enabled per-request, this bounds the runaway 400-900-token
#    reasoning, and the end-of-thinking injection avoids tool-call truncation.
# REMOVED --dry-multiplier / --repeat-penalty: they corrupt verbatim copying of
#    repeated-token strings. A/B proved it — UUIDs copied 0/3 WITH them, 3/3
#    WITHOUT. Repetition penalties punish the repeated hex digits in UUIDs/hashes/
#    SHAs/tokens, forcing the model to swap characters. Net-negative: proven harm
#    (silent ID/hash/token corruption) for an unproven runaway benefit. DO NOT
#    re-add. If the output runaway recurs, fix it some way that doesn't break
#    verbatim output.
# NOTE: --cache-reuse was tried and removed — llama.cpp disables it for multimodal
# models ("cache_reuse is not supported by multimodal"), and we load --mmproj for
# computer-use vision. So the mid-run prefix-divergence cache-misses can't be fixed
# this way while vision is needed. Don't re-add it.
REQUIRED_KV=("--reasoning-budget 256")

if [[ ! -f "$PLIST" ]]; then
  echo "llama plist not found at: $PLIST (nothing to do)"
  exit 0
fi

pb() { /usr/libexec/PlistBuddy -c "$1" "$PLIST"; }

changed=0
backup_once() {
  if [[ $changed -eq 0 ]]; then
    cp "$PLIST" "$PLIST.bak.ensureflags.$(date +%s)"
  fi
}

for flag in "${REQUIRED_FLAGS[@]}"; do
  if pb "Print :ProgramArguments" | grep -q -- "$flag"; then
    echo "present: $flag"
    continue
  fi
  backup_once
  pb "Add :ProgramArguments: string $flag"
  echo "added:   $flag"
  changed=1
done

for pair in "${REQUIRED_KV[@]}"; do
  key="${pair%% *}"
  val="${pair#* }"
  if pb "Print :ProgramArguments" | grep -q -- "$key"; then
    echo "present: $key"
    continue
  fi
  backup_once
  pb "Add :ProgramArguments: string $key"
  pb "Add :ProgramArguments: string $val"
  echo "added:   $key $val"
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
