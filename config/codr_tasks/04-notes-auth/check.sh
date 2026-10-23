set -u
[ -f app.py ] || { echo "FAIL: app.py missing"; exit 1; }
[ -f ../03-notes-api/app.py ] || { echo "FAIL: ../03-notes-api/app.py missing (the task-03 app is the migration source)"; exit 1; }
python3 -I - <<'PY'
import json, os, shutil, socket, sqlite3, subprocess, sys, tempfile, time, urllib.error, urllib.request

# `-I` keeps the task folder off this interpreter's path; nothing at the top
# level may shadow the standard library for the child Pythons (DP-449 review).
_shadows = sorted(n for n in os.listdir(".") if n.removesuffix(".py") in sys.stdlib_module_names)
if _shadows:
    print(f"FAIL: these names shadow Python's standard library: {', '.join(_shadows)}")
    sys.exit(1)

failures = []
tmp = tempfile.mkdtemp()
# The task-03 app runs from a private copy, so nothing else in its folder matters.
old_app = os.path.join(tmp, "old_app.py")
shutil.copy("../03-notes-api/app.py", old_app)

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def start(app, port, db, env=None):
    proc = subprocess.Popen([sys.executable, app, "--port", str(port), "--db", db],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                            env={**os.environ, **(env or {})})
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return proc
        except OSError:
            if proc.poll() is not None:
                break
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError(f"{app} did not start: {proc.stderr.read()[-400:] if proc.stderr else ''}")

def stop(proc):
    proc.kill()
    proc.wait()

def call(port, method, path, body=None, token=None):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()

def js(b):
    try:
        return json.loads(b)
    except ValueError:
        return None

def expect(cond, msg):
    if not cond:
        failures.append(msg)

def texts(port, token):
    st, b = call(port, "GET", "/notes", token=token)
    return st, [n.get("text") for n in (js(b) or []) if isinstance(n, dict)]

def login(port, user, pw):
    st, b = call(port, "POST", "/login", {"username": user, "password": pw})
    return st, (js(b) or {}).get("token")

# 1. A task-03 database with two notes.
db = os.path.join(tmp, "notes.db")
port = free_port()
try:
    proc = start(old_app, port, db)
except RuntimeError as e:
    print(f"FAIL: task-03 app: {e}")
    sys.exit(1)
try:
    for t in ("old one", "old two"):
        st, _ = call(port, "POST", "/notes", {"text": t})
        if st != 201:
            print(f"FAIL: the task-03 app could not create a note ({st}), so there is no database to migrate")
            sys.exit(1)
finally:
    stop(proc)

# 2. The new app upgrades it.
port = free_port()
try:
    proc = start("app.py", port, db, {"NOTES_LEGACY_PASSWORD": "s3cret-legacy"})
except RuntimeError as e:
    print(f"FAIL: app.py on a task-03 database: {e}")
    sys.exit(1)
try:
    st, _ = call(port, "GET", "/notes")
    expect(st == 401, f"GET /notes without a token: expected 401, got {st}")
    st, _ = call(port, "GET", "/notes", token="not-a-real-token")
    expect(st == 401, f"GET /notes with a bad token: expected 401, got {st}")

    st, tl = login(port, "legacy", "s3cret-legacy")
    expect(st == 200 and bool(tl), f"login as legacy with NOTES_LEGACY_PASSWORD: {st}")
    if tl:
        st, t = texts(port, tl)
        expect(st == 200 and t == ["old one", "old two"], f"legacy's notes after the upgrade: {st} {t}")
    st, _ = login(port, "legacy", "legacy")
    expect(st == 401, f"login as legacy with the default password while the env var is set: expected 401, got {st}")

    st, b = call(port, "POST", "/users", {"username": "alice", "password": "pw-alice-123"})
    u = js(b) or {}
    expect(st == 201 and u.get("username") == "alice" and isinstance(u.get("id"), int), f"POST /users alice: {st} {b[:200]!r}")
    st, b = call(port, "POST", "/users", {"username": "alice", "password": "other"})
    expect(st == 409 and "error" in (js(b) or {}), f"duplicate username: expected 409 + error, got {st}")
    for bad, why in (({"username": "bob"}, "missing password"), ({"username": "", "password": "x"}, "empty username")):
        st, b = call(port, "POST", "/users", bad)
        expect(st == 400 and "error" in (js(b) or {}), f"POST /users {why}: expected 400 + error, got {st}")
    call(port, "POST", "/users", {"username": "bob", "password": "pw-bob-456"})

    st, _ = login(port, "alice", "wrong")
    expect(st == 401, f"login with a wrong password: expected 401, got {st}")
    st, _ = login(port, "nobody", "x")
    expect(st == 401, f"login as an unknown user: expected 401, got {st}")
    _, ta = login(port, "alice", "pw-alice-123")
    _, tb = login(port, "bob", "pw-bob-456")
    expect(bool(ta and tb), "alice and bob could not log in")
    if ta and tb:
        st, b = call(port, "POST", "/notes", {"text": "alice private"}, token=ta)
        an = js(b) or {}
        expect(st == 201 and an.get("text") == "alice private", f"POST /notes as alice: {st} {b[:200]!r}")
        call(port, "POST", "/notes", {"text": "bob private"}, token=tb)
        expect(texts(port, ta)[1] == ["alice private"], f"alice sees: {texts(port, ta)[1]}")
        expect(texts(port, tb)[1] == ["bob private"], f"bob sees: {texts(port, tb)[1]}")
        nid = an.get("id")
        st, _ = call(port, "GET", f"/notes/{nid}", token=tb)
        expect(st == 404, f"bob GETs alice's note: expected 404, got {st}")
        st, _ = call(port, "PUT", f"/notes/{nid}", {"text": "hacked"}, token=tb)
        expect(st == 404, f"bob PUTs alice's note: expected 404, got {st}")
        st, _ = call(port, "DELETE", f"/notes/{nid}", token=tb)
        expect(st == 404, f"bob DELETEs alice's note: expected 404, got {st}")
        st, b = call(port, "GET", f"/notes/{nid}", token=ta)
        expect(st == 200 and (js(b) or {}).get("text") == "alice private", f"alice's note after bob's tries: {st} {b[:120]!r}")
        st, b = call(port, "GET", "/notes?q=PRIVATE", token=ta)
        expect(st == 200 and [n.get("text") for n in (js(b) or [])] == ["alice private"], f"alice ?q=PRIVATE: {st} {b[:200]!r}")
        st, b = call(port, "PUT", f"/notes/{nid}", {"text": "alice edited"}, token=ta)
        expect(st == 200 and (js(b) or {}).get("text") == "alice edited", f"alice PUT: {st} {b[:200]!r}")
        st, _ = call(port, "POST", "/notes", {"text": ""}, token=ta)
        expect(st == 400, f"POST empty text with a token: expected 400, got {st}")
finally:
    stop(proc)

# 3. No password is stored in plain text.
con = sqlite3.connect(db)
plain = set()
for (table,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
    for row in con.execute(f'SELECT * FROM "{table}"'):
        for cell in row:
            if isinstance(cell, (str, bytes)):
                s = cell if isinstance(cell, str) else cell.decode("latin-1")
                plain |= {pw for pw in ("pw-alice-123", "pw-bob-456", "s3cret-legacy") if pw in s}
con.close()
expect(not plain, f"passwords stored in plain text: {sorted(plain)}")

# 4. A second start on the upgraded database changes nothing.
port = free_port()
try:
    proc = start("app.py", port, db)
    try:
        _, tl = login(port, "legacy", "s3cret-legacy")
        expect(bool(tl) and texts(port, tl)[1] == ["old one", "old two"], "legacy after a second start: login or notes lost")
        _, ta = login(port, "alice", "pw-alice-123")
        expect(bool(ta) and texts(port, ta)[1] == ["alice edited"], "alice after a second start: login or notes lost")
    finally:
        stop(proc)
except RuntimeError as e:
    failures.append(f"second start on the upgraded database: {e}")

# 5. A fresh database works too.
port = free_port()
try:
    proc = start("app.py", port, os.path.join(tmp, "fresh.db"))
    try:
        st, _ = call(port, "POST", "/users", {"username": "carol", "password": "pw-carol"})
        _, tc = login(port, "carol", "pw-carol")
        expect(st == 201 and bool(tc) and texts(port, tc) == (200, []),
               "fresh database: carol can't sign up, log in, or list an empty notes list")
    finally:
        stop(proc)
except RuntimeError as e:
    failures.append(f"fresh database: {e}")

r = subprocess.run([sys.executable, "-m", "unittest"], capture_output=True, text=True, timeout=300)
expect(r.returncode == 0 and "Ran 0 tests" not in r.stderr, f"python3 -m unittest: exit {r.returncode}: {r.stderr[-400:]}")

print("\n".join(f"FAIL: {f}" for f in failures) or "PASS")
sys.exit(1 if failures else 0)
PY
