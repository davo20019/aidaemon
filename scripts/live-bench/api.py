import json, os, urllib.request, uuid, time
BASE = os.environ.get("AIDAEMON_GATEWAY", "http://127.0.0.1:8788/client/v1")
TOKEN = open(os.path.expanduser("~/.aidaemon/client-token")).read().strip()
def _req(method, path, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method,
        headers={"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read() or b"{}")
def cmd(command):
    return _req("POST", "/commands", {"type": "command", "request_id": str(uuid.uuid4()), "command": command})
def events(since=None, wait=20):
    q = f"?wait={wait}" + (f"&since={since}" if since is not None else "")
    return _req("GET", "/events" + q, timeout=wait + 15)
