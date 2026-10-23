set -u
[ -f csvstats.py ] || { echo "FAIL: csvstats.py missing"; exit 1; }
python3 -I - <<'PY'
import json, math, os, statistics, subprocess, sys, tempfile

# Hidden-check hardening (DP-449 review): `-I` keeps the task folder off this
# interpreter's path, and nothing at its top level may shadow the standard
# library for the child Pythons the check runs from it (`-m unittest` puts the
# working directory first on their path).
_shadows = sorted(n for n in os.listdir(".") if n.removesuffix(".py") in sys.stdlib_module_names)
if _shadows:
    print(f"FAIL: these names shadow Python's standard library: {', '.join(_shadows)}")
    sys.exit(1)

failures = []

def run(*args):
    return subprocess.run([sys.executable, "csvstats.py", *args], capture_output=True, text=True, timeout=60)

def close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)

rows = [
    ["id", "price", "qty", "name", "ratio"],
    ["1", "10.5", "3", "apple", "0.5"],
    ["2", "", "4", "pear", "-1e2"],
    ["3", "7.25", "", "fig", "2"],
    ["4", "100", "1", "kiwi", "3.75"],
]
d = tempfile.mkdtemp()
path = os.path.join(d, "data.csv")
with open(path, "w") as f:
    f.write("\n".join(",".join(r) for r in rows) + "\n")

def expected(col):
    i = rows[0].index(col)
    vals = [float(r[i]) for r in rows[1:] if r[i] != ""]
    return {"count": len(vals), "mean": statistics.mean(vals), "min": min(vals), "max": max(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else None}

r = run(path, "--format", "json")
if r.returncode != 0:
    failures.append(f"--format json exited {r.returncode}: {r.stderr[-500:]}")
else:
    try:
        got = json.loads(r.stdout)
    except ValueError:
        got = None
        failures.append(f"--format json is not one JSON object: {r.stdout[:300]!r}")
    if got is not None:
        if list(got) != ["id", "price", "qty", "ratio"]:
            failures.append(f"columns: expected ['id', 'price', 'qty', 'ratio'], got {list(got)}")
        for col in ["id", "price", "qty", "ratio"]:
            if col not in got:
                continue
            for key, want in expected(col).items():
                if not close(got[col].get(key), want):
                    failures.append(f"{col}.{key}: expected {want}, got {got[col].get(key)}")

r = run(path, "--columns", "ratio,qty", "--format", "json")
if r.returncode != 0 or list(json.loads(r.stdout or "{}")) != ["ratio", "qty"]:
    failures.append(f"--columns ratio,qty: exit {r.returncode}, out {r.stdout[:200]!r}")

for args, why in (([path, "--columns", "name"], "non-numeric named column"),
                  ([path, "--columns", "nope"], "missing named column"),
                  ([os.path.join(d, "absent.csv")], "missing file")):
    r = run(*args)
    if r.returncode != 2:
        failures.append(f"{why}: expected exit 2, got {r.returncode}")

r = run(path)
if r.returncode != 0 or "price" not in r.stdout:
    failures.append(f"default table output: exit {r.returncode}, out {r.stdout[:200]!r}")
r = run("--help")
if r.returncode != 0 or "--columns" not in r.stdout:
    failures.append("--help does not describe --columns")

r = subprocess.run([sys.executable, "-m", "unittest"], capture_output=True, text=True, timeout=300)
if r.returncode != 0 or "Ran 0 tests" in r.stderr:
    failures.append(f"python3 -m unittest: exit {r.returncode}: {r.stderr[-400:]}")

print("\n".join(f"FAIL: {f}" for f in failures) or "PASS")
sys.exit(1 if failures else 0)
PY
