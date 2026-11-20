#!/bin/bash
# usage: q.sh "SQL" — read-only query against the live encrypted DB
# (AIDAEMON_ENCRYPTION_KEY from the repo's .env; AIDAEMON_DB_PATH or aidaemon.db).
cd "$(dirname "$0")/../.." || exit 1
KEY=$(grep '^AIDAEMON_ENCRYPTION_KEY=' .env | head -1 | cut -d= -f2- | tr -d '"'"'"'')
sqlcipher -readonly "${AIDAEMON_DB_PATH:-aidaemon.db}" "PRAGMA key='$KEY';" ".mode list" "$1" 2>&1 | grep -v '^ok$'
