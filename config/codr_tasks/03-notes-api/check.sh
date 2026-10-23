set -u
[ -f app.py ] || { echo "FAIL: app.py missing"; exit 1; }
python3 -I - <<'PY'
import json, os, socket, subprocess, sys, tempfile, time, urllib.error, urllib.request

# Hidden-check hardening (DP-449 review): `-I` keeps the task folder off this
# interpreter's path, and nothing at its top level may shadow the standard
# library for the child Pythons the check runs from it (`-m unittest` puts the
# working directory first on their path).
_shadows = sorted(n for n in os.listdir(".") if n.removesuffix(".py") in sys.stdlib_module_names)
if _shadows:
    print(f"FAIL: these names shadow Python's standard library: {', '.join(_shadows)}")
    sys.exit(1)

failures = []

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def start(port, db):
    proc = subprocess.Popen([sys.executable, "app.py", "--port", str(port), "--db", db],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return proc
        except OSError:
            if proc.poll() is not None:
                break
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError(f"server did not start: {proc.stderr.read()[-400:] if proc.stderr else ''}")

def call(port, method, path, body=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type", "")

def js(b):
    try:
        return json.loads(b)
    except ValueError:
        return None

def expect(cond, msg):
    if not cond:
        failures.append(msg)

db = os.path.join(tempfile.mkdtemp(), "check.db")
port = free_port()
try:
    proc = start(port, db)
except RuntimeError as e:
    print(f"FAIL: {e}")
    sys.exit(1)
try:
    st, b, _ = call(port, "POST", "/notes", {"text": "Buy Milk"})
    a = js(b) or {}
    expect(st == 201 and a.get("text") == "Buy Milk" and isinstance(a.get("id"), int) and a.get("created"),
           f"POST /notes: {st} {b[:200]!r}")
    st, b, _ = call(port, "POST", "/notes", {"text": "call bob"})
    second = js(b) or {}
    for bad, why in (({"text": ""}, "empty text"), ({}, "missing text")):
        st, b, _ = call(port, "POST", "/notes", bad)
        expect(st == 400 and "error" in (js(b) or {}), f"POST {why}: expected 400 + error, got {st} {b[:120]!r}")
    st, b, _ = call(port, "POST", "/notes", raw=b"not json")
    expect(st == 400, f"POST non-JSON body: expected 400, got {st}")

    st, b, _ = call(port, "GET", "/notes")
    lst = js(b)
    expect(st == 200 and isinstance(lst, list) and [n.get("text") for n in lst] == ["Buy Milk", "call bob"],
           f"GET /notes: {st} {b[:200]!r}")
    st, b, _ = call(port, "GET", "/notes?q=milk")
    expect(st == 200 and [n.get("text") for n in (js(b) or [])] == ["Buy Milk"], f"GET ?q=milk: {st} {b[:200]!r}")

    st, b, _ = call(port, "GET", f"/notes/{a.get('id')}")
    expect(st == 200 and (js(b) or {}).get("text") == "Buy Milk", f"GET /notes/<id>: {st}")
    st, b, _ = call(port, "GET", "/notes/999999")
    expect(st == 404 and "error" in (js(b) or {}), f"GET unknown id: expected 404 + error, got {st}")

    st, b, _ = call(port, "PUT", f"/notes/{a.get('id')}", {"text": "Buy oat milk"})
    u = js(b) or {}
    expect(st == 200 and u.get("text") == "Buy oat milk" and u.get("id") == a.get("id")
           and u.get("created") == a.get("created"), f"PUT: {st} {b[:200]!r}")
    st, _, _ = call(port, "PUT", "/notes/999999", {"text": "x"})
    expect(st == 404, f"PUT unknown id: expected 404, got {st}")
    st, _, _ = call(port, "PUT", f"/notes/{a.get('id')}", {"text": ""})
    expect(st == 400, f"PUT empty text: expected 400, got {st}")

    st, b, _ = call(port, "DELETE", f"/notes/{second.get('id')}")
    expect(st == 204 and not b, f"DELETE: expected 204 and no body, got {st} {b[:80]!r}")
    st, _, _ = call(port, "DELETE", f"/notes/{second.get('id')}")
    expect(st == 404, f"DELETE again: expected 404, got {st}")

    st, b, ctype = call(port, "GET", "/")
    expect(st == 200 and "html" in ctype.lower() and b"<form" in b.lower() and b"oat milk" in b.lower(),
           f"GET /: {st} {ctype} (needs an HTML page listing notes with a form)")
finally:
    proc.kill()
    proc.wait()

# Persistence across a restart.
port = free_port()
try:
    proc = start(port, db)
    st, b, _ = call(port, "GET", "/notes")
    expect([n.get("text") for n in (js(b) or [])] == ["Buy oat milk"], f"after restart: {b[:200]!r}")
    proc.kill()
    proc.wait()
except RuntimeError as e:
    failures.append(f"restart: {e}")

r = subprocess.run([sys.executable, "-m", "unittest"], capture_output=True, text=True, timeout=300)
expect(r.returncode == 0 and "Ran 0 tests" not in r.stderr, f"python3 -m unittest: exit {r.returncode}: {r.stderr[-400:]}")

print("\n".join(f"FAIL: {f}" for f in failures) or "PASS")
sys.exit(1 if failures else 0)
PY
