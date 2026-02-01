---
name: system-admin
description: System diagnostics, monitoring, and administration tasks
triggers: disk, memory, cpu, process, docker, service, systemctl, logs, restart, kill, port, network
---
When the user asks about system administration tasks, follow a diagnostics-first approach.

### Diagnostics First
Always gather information before taking action:
1. Understand what the user is reporting or asking about.
2. Run diagnostic commands to confirm the situation.
3. Explain what you found.
4. Propose a fix and execute it (or ask for approval if destructive).

### Common Patterns

**Disk Space:**
- `df -h` — overview of all filesystems
- `du -sh /path/*` — breakdown of a specific directory
- `find /path -type f -size +100M` — find large files

**Memory:**
- `free -h` — memory overview
- `ps aux --sort=-%mem | head -20` — top memory consumers

**CPU:**
- `top -bn1 | head -20` — CPU snapshot
- `ps aux --sort=-%cpu | head -20` — top CPU consumers

**Docker:**
- `docker ps` — running containers
- `docker logs <container>` — container logs
- `docker stats --no-stream` — resource usage

**Services (systemd):**
- `systemctl status <service>` — service status
- `journalctl -u <service> --since "1 hour ago"` — recent logs
- `systemctl restart <service>` — restart (ask user first)

**Network:**
- `ss -tlnp` or `netstat -tlnp` — listening ports
- `ping -c 3 <host>` — connectivity check
- `curl -I <url>` — HTTP health check

### Safety Rules
- NEVER run `rm -rf` on system directories without explicit user confirmation.
- NEVER restart critical services (sshd, networking) without warning the user.
- Always show the user what you found before taking corrective action.
- Prefer non-destructive commands first (status, list, check) before modifying anything.
