"""Delete what a live-bench run wrote into the live daemon's memory.
usage: cleanup.py <since, e.g. 2026-11-20T04:10> [--dry]
The run's sessions are the app sessions since then whose first message is a
bench prompt; their events, episodes, spans and the facts, procedures and
error solutions created since then go, with the hbench projects and their
notes. Consolidation re-extracts from what is left, so delete it all."""
import subprocess, sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
Q = os.path.join(ROOT, "scripts/live-bench/q.sh")
W = sys.argv[1]
dry = "--dry" in sys.argv
def q(sql):
    out = subprocess.run([Q, sql], capture_output=True, text=True).stdout.strip()
    return [l.split("|") for l in out.splitlines() if l]
PROMPTS = ["hbench", "capital of Australia", "17 * 23", "Send the report to him now", "bike locker",
           "gym class", "grocery list for pancakes", "add one more item for a topping", "Scroll test",
           "The tests fail. Find the bug and fix it.", "Make slugify drop punctuation", "Add f_to_c(f) to temps.py",
           "Rename fetch_items to load_items", "gym locker combination", "synthetic test pottery"]
like = " or ".join(f"instr(data, '{p}') > 0" for p in PROMPTS)
sids = [r[0] for r in q(f"select distinct session_id from events where event_type='user_message' and created_at >= '{W}' and session_id like 'app:%' and ({like})")]
ins = ",".join(f"'{s}'" for s in sids) or "''"
markers = ["gym", "locker", "synthetic_test", "hbench", "4417"]
mk = " or ".join(f"instr(lower(key||' '||value), '{m}') > 0" for m in markers)
fact_ids = [r[0] for r in q(f"select id from facts where created_at >= '{W}' and (({mk}) or channel_id in ({ins}))")]
ep_ids = [r[0] for r in q(f"select id from episodes where session_id in ({ins})")]
span_ids = [r[0] for r in q(f"select id from memory_spans where session_id in ({ins})")]
proc_ids = [r[0] for r in q(f"select id from procedures where created_at >= '{W}'")]
err_ids = [r[0] for r in q(f"select id from error_solutions where created_at >= '{W}'")]
proj_ids = [r[0] for r in q("select id from work_projects where name like 'hbench %'")]
print(f"sessions {len(sids)} facts {len(fact_ids)} episodes {len(ep_ids)} spans {len(span_ids)} procedures {len(proc_ids)} errors {len(err_ids)} projects {len(proj_ids)}")
def lst(xs, quote=False):
    return ",".join((f"'{x}'" if quote else x) for x in xs) or ("''" if quote else "-1")
stmts = [
    f"DELETE FROM memory_embeddings WHERE owner_type IN ('fact','fact_key') AND owner_id IN ({lst(fact_ids)});",
    f"DELETE FROM memory_embeddings WHERE owner_type = 'episode' AND owner_id IN ({lst(ep_ids)});",
    f"DELETE FROM memory_embeddings WHERE owner_type = 'span' AND owner_id IN ({lst(span_ids)});",
    f"DELETE FROM memory_embeddings WHERE owner_type = 'procedure' AND owner_id IN ({lst(proc_ids)});",
    f"DELETE FROM memory_embeddings WHERE owner_type = 'error_solution' AND owner_id IN ({lst(err_ids)});",
    f"DELETE FROM facts WHERE id IN ({lst(fact_ids)});",
    f"DELETE FROM procedures WHERE id IN ({lst(proc_ids)});",
    f"DELETE FROM error_solutions WHERE id IN ({lst(err_ids)});",
    f"DELETE FROM session_work_projects WHERE project_id IN ({lst(proj_ids, True)}) OR session_id IN ({ins});",
    f"DELETE FROM work_projects WHERE id IN ({lst(proj_ids, True)});",
]
for t in ["episodes", "memory_spans", "events", "task_plans", "cli_agent_invocations", "notification_queue",
          "conversation_summaries", "session_channels", "session_context_boundaries", "history_message_index",
          "filesystem_checkpoints", "checkpoint_restore_runs", "task_lifecycle", "background_spawns",
          "channel_approvals", "client_events"]:
    n = q(f"select count(*) from {t} where session_id in ({ins})")
    print(f"  {t}: {n[0][0] if n else '?'}")
    stmts.append(f"DELETE FROM {t} WHERE session_id IN ({ins});")
if dry:
    sys.exit(0)
key = [l.split("=", 1)[1].strip().strip("'\"") for l in open(os.path.join(ROOT, ".env")) if l.startswith("AIDAEMON_ENCRYPTION_KEY=")][0]
sql = "BEGIN IMMEDIATE;\n" + "\n".join(stmts) + "\nCOMMIT;"
p = subprocess.run(["sqlcipher", "aidaemon.db", f"PRAGMA key='{key}';", "PRAGMA busy_timeout=10000;", sql], cwd=ROOT, capture_output=True, text=True)
err = [l for l in (p.stdout + p.stderr).splitlines() if l not in ("ok", "10000")]
print("deleted" if not err else "ERROR: " + " ".join(err)[:500])
for pid in proj_ids:
    d = os.path.expanduser(f"~/.aidaemon/projects/{pid}")
    if os.path.isdir(d):
        import shutil; shutil.rmtree(d); print("removed notes", d)
