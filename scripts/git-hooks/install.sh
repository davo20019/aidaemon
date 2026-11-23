#!/bin/sh
# Install the pre-push gate for this clone. The installed hook runs the
# pushing worktree's own scripts/git-hooks/pre-push, so every worktree checks
# with its own branch's copy.
set -e
hook="$(git rev-parse --git-common-dir)/hooks/pre-push"
rm -f "$hook"
cat > "$hook" <<'HOOK'
#!/bin/sh
# Runs the pushing worktree's own hook (scripts/git-hooks/pre-push).
exec "$(git rev-parse --show-toplevel)/scripts/git-hooks/pre-push" "$@"
HOOK
chmod +x "$hook"
echo "Installed $hook"
