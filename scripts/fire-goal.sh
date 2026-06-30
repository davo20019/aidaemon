#!/usr/bin/env bash
# Fire an aidaemon goal NOW.
#
# Re-arms the goal (active, today's budget reset, failure counter cleared) and
# inserts a one-shot schedule that is already due, so the running daemon picks
# it up on its next heartbeat tick (~30s) — no restart needed.
#
# Usage:
#   ./scripts/fire-goal.sh <goal-id-prefix>
# Examples:
#   ./scripts/fire-goal.sh 9a744834   # the daily tweet goal
#   ./scripts/fire-goal.sh 4a308b23   # the blog goal
set -euo pipefail

PREFIX="${1:?usage: fire-goal.sh <goal-id-prefix>}"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
KEY="$(grep -E '^AIDAEMON_ENCRYPTION_KEY=' "$DIR/.env" | head -1 | cut -d= -f2- | tr -d "\"' ")"
DB="${AIDAEMON_DB_PATH:-$DIR/aidaemon.db}"
SID="$(uuidgen)"
NOW="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
# 5 minutes in the past so the schedule is unambiguously due (BSD/macOS, then GNU).
PAST="$(date -u -v-5M +%Y-%m-%dT%H:%M:%S+00:00 2>/dev/null || date -u -d '5 minutes ago' +%Y-%m-%dT%H:%M:%S+00:00)"

sqlcipher "$DB" <<SQL
PRAGMA key='${KEY}';
PRAGMA busy_timeout=8000;
UPDATE goals
   SET status='active', tokens_used_today=0, dispatch_failures=0, completed_at=NULL, updated_at='${NOW}'
 WHERE id LIKE '${PREFIX}%';
INSERT INTO goal_schedules (id, goal_id, cron_expr, tz, fire_policy, is_one_shot, is_paused, next_run_at, created_at, updated_at)
SELECT '${SID}', id, '* * * * *', 'local', 'coalesce', 1, 0, '${PAST}', '${NOW}', '${NOW}'
  FROM goals WHERE id LIKE '${PREFIX}%';
SELECT 'fired '||substr(id,1,8)||' ('||status||'): '||substr(description,1,50) FROM goals WHERE id LIKE '${PREFIX}%';
SQL

echo "Re-armed ${PREFIX}. The daemon will dispatch it within ~30s. Watch:"
echo "  tail -f ~/Library/Logs/aidaemon/stdout.log"
