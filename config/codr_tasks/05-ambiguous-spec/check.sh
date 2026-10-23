set -u
[ -f wordfreq.py ] || { echo "FAIL: wordfreq.py missing"; exit 1; }
python3 -I - <<'PY'
# Grades only what every reasonable reading of the spec agrees on: plain
# ASCII words split by spaces and newlines, distinct counts (no ties), no
# punctuation. Word rules and tie order are codr's to choose.
import json, os, subprocess, sys, tempfile

_shadows = sorted(n for n in os.listdir(".") if n.removesuffix(".py") in sys.stdlib_module_names)
if _shadows:
    print(f"FAIL: these names shadow Python's standard library: {', '.join(_shadows)}")
    sys.exit(1)

failures = []

def expect(cond, msg):
    if not cond:
        failures.append(msg)

def run(*args, stdin=None):
    return subprocess.run([sys.executable, "wordfreq.py", *args], input=stdin,
                          capture_output=True, text=True, timeout=60)

def pairs(out):
    rows = []
    for line in out.strip().splitlines():
        parts = line.split()
        if len(parts) != 2 or not parts[1].isdigit():
            return None
        rows.append((parts[0], int(parts[1])))
    return rows

counts = {"alpha": 7, "bravo": 5, "charlie": 4, "delta": 3, "echo": 2, "foxtrot": 1}
words = []
for w, c in counts.items():
    words += [w.upper() if i % 3 == 0 else w for i in range(c)]
words = words[::2] + words[1::2]          # interleave so order of appearance != rank
text = " ".join(words[:10]) + "\n" + " ".join(words[10:]) + "\n"
expected = sorted(counts.items(), key=lambda kv: -kv[1])

tmp = tempfile.mkdtemp()
path = os.path.join(tmp, "a.txt")
with open(path, "w") as f:
    f.write(text)

r = run(path)
expect(r.returncode == 0 and pairs(r.stdout) == expected, f"default on a file: exit {r.returncode}, got {r.stdout[:300]!r}")
r = run(path, "-n", "3")
expect(r.returncode == 0 and pairs(r.stdout) == expected[:3], f"-n 3: got {r.stdout[:200]!r}")
r = run("-n", "2", stdin=text)
expect(r.returncode == 0 and pairs(r.stdout) == expected[:2], f"stdin with -n 2: got {r.stdout[:200]!r}")
r = run(path, "--json")
try:
    got = [tuple(p) for p in json.loads(r.stdout)]
except (ValueError, TypeError):
    got = None
expect(r.returncode == 0 and got == expected, f"--json: got {r.stdout[:300]!r}")

path2 = os.path.join(tmp, "b.txt")
with open(path2, "w") as f:
    f.write("foxtrot foxtrot foxtrot foxtrot foxtrot foxtrot foxtrot foxtrot\n")
r = run(path, path2, "-n", "1")
expect(r.returncode == 0 and pairs(r.stdout) == [("foxtrot", 9)], f"two files are counted together: got {r.stdout[:200]!r}")

r = run(os.path.join(tmp, "missing.txt"))
expect(r.returncode == 2 and r.stderr.strip(), f"missing file: expected exit 2 + stderr, got {r.returncode}")

expect(os.path.isfile("DECISIONS.md") and len(open("DECISIONS.md").read().split()) >= 20,
       "DECISIONS.md is missing or nearly empty")

r = subprocess.run([sys.executable, "-m", "unittest"], capture_output=True, text=True, timeout=300)
expect(r.returncode == 0 and "Ran 0 tests" not in r.stderr, f"python3 -m unittest: exit {r.returncode}: {r.stderr[-400:]}")

print("\n".join(f"FAIL: {f}" for f in failures) or "PASS")
sys.exit(1 if failures else 0)
PY
