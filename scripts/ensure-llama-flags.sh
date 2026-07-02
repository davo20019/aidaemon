#!/usr/bin/env bash
# ensure-llama-flags.sh — idempotently ensure the local llama-server launchd job
# carries the flags aidaemon depends on. Safe to run repeatedly; a no-op when
# the flags are already present.
#
# Currently required: --mlock (plus --reasoning-budget below).
#
# REMOVED --kv-unified (2026-07-02): it was added to enable the idle-slot RAM
# cache (--cache-idle-slots) for goal runs, but live measurement showed that
# cache is write-only in practice: 306 "saving idle slot" events and ZERO
# restores over the server's entire 2-day lifetime. Worse, the idle-save FREES
# the slot's live KV, so once conversation prompts grew past ~14K tokens the
# interactive slot's warm prefix was destroyed between every user prompt —
# each one paid a 36-66s cold prefill (cached=0 on every task start from
# 15:32Z on 2026-07-02). Without --kv-unified each slot keeps its own KV pool:
# verified by direct A/B (big slot-0 prompt stayed 99.9% cached across three
# 15K-token slot-1 jobs, prefill 36s -> 0.2s). Cost: llama-server RSS grows
# (~17.6 -> ~32 GB, full per-slot KV preallocation) and slot-1 users (goal
# runs vs memory pipeline) once again evict each other — watch goal-run
# cached_input_tokens; if that regresses badly the fix is scheduling, not
# re-adding --kv-unified. DO NOT re-add without confirming the restore path
# actually loads (grep the llama log for a load counterpart to
# "saving idle slot").
set -euo pipefail

PLIST="${LLAMA_PLIST:-$HOME/Library/LaunchAgents/ai.aidaemon.llama.plist}"
# Bare flags (no value).
#  --mlock: pin the model in RAM so macOS can't compress/swap it under memory
#    pressure — avoids latency spikes (steadier tail latency). Non-fatal if it
#    can't lock (llama.cpp warns and continues).
REQUIRED_FLAGS=("--mlock")
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
