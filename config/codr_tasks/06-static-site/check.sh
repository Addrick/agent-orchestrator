set -u
[ -f ssg.py ] || { echo "FAIL: ssg.py missing"; exit 1; }
python3 -I - <<'PY'
import os, re, shutil, subprocess, sys, tempfile, time

_shadows = sorted(n for n in os.listdir(".") if n.removesuffix(".py") in sys.stdlib_module_names)
if _shadows:
    print(f"FAIL: these names shadow Python's standard library: {', '.join(_shadows)}")
    sys.exit(1)

failures = []

def expect(cond, msg):
    if not cond:
        failures.append(msg)

def norm(html):
    """Whitespace between tags and at the ends does not matter."""
    return re.sub(r">\s+<", "><", html.strip())

def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()

def build(src, out):
    return subprocess.run([sys.executable, os.path.abspath("ssg.py"), src, out],
                          capture_output=True, text=True, timeout=120)

def counts(r):
    lines = r.stdout.strip().splitlines()
    m = re.fullmatch(r"built (\d+), skipped (\d+), removed (\d+)", lines[-1].strip()) if lines else None
    return tuple(int(g) for g in m.groups()) if m else None

tmp = tempfile.mkdtemp()
src, out = os.path.join(tmp, "src"), os.path.join(tmp, "out")
write(f"{src}/templates/post.html", "<html><title>{{ title }}</title><body data-slug=\"{{slug}}\" data-x=\"{{ nope }}\">\n<p class=\"meta\">{{ author }} {{ date }}</p>\n{{ content }}\n</body></html>\n")
write(f"{src}/templates/index.html", "<ul>\n{{ posts }}\n</ul>\n")

write(f"{src}/content/hello.md", """---
title: Hello & <World>
date: 2026-01-05
author: Ann: the "first"
---
# Hello *there*

This is **bold** and *em* and `a <b> & *c*` code,
continued on a second line with a [link](https://example.com/x?a=1).

- one
- two **2**

1. first
2. second

```python
if a < b:

    print("*not em*")
```

## Sub & head
""")
write(f"{src}/content/second.md", "---\ntitle: Second\ndate: 2026-03-01\n---\nJust text.\n")
write(f"{src}/content/alpha.md", "---\ntitle: Alpha\ndate: 2026-03-01\n---\nSame date as second.\n")
write(f"{src}/content/undated.md", "---\ntitle: No date\n---\nUndated.\n")

r = build(src, out)
expect(r.returncode == 0, f"first build: exit {r.returncode}: {r.stderr[-400:]}")
expect(counts(r) == (5, 0, 0), f"first build summary: {r.stdout.strip().splitlines()[-1:] }")

want_hello = norm("""<html><title>Hello &amp; &lt;World&gt;</title><body data-slug="hello" data-x="">
<p class="meta">Ann: the "first" 2026-01-05</p>
<h1>Hello <em>there</em></h1>
<p>This is <strong>bold</strong> and <em>em</em> and <code>a &lt;b&gt; &amp; *c*</code> code, continued on a second line with a <a href="https://example.com/x?a=1">link</a>.</p>
<ul><li>one</li><li>two <strong>2</strong></li></ul>
<ol><li>first</li><li>second</li></ol>
<pre><code>if a &lt; b:

    print("*not em*")</code></pre>
<h2>Sub &amp; head</h2>
</body></html>""")
try:
    got = norm(read(f"{out}/hello.html"))
    # The meta line's quotes may be escaped or not; both are correct HTML.
    got = got.replace("&quot;", '"').replace("&#x27;", "'").replace("&#39;", "'")
    # A trailing newline inside <pre><code> is allowed.
    got = got.replace('")\n</code></pre>', '")</code></pre>')
    expect(got == want_hello, f"hello.html differs.\n  want: {want_hello}\n  got:  {got}")
except OSError as e:
    failures.append(f"hello.html: {e}")

want_index = norm("""<ul>
<li><a href="alpha.html">Alpha</a></li>
<li><a href="second.html">Second</a></li>
<li><a href="hello.html">Hello &amp; &lt;World&gt;</a></li>
<li><a href="undated.html">No date</a></li>
</ul>""")
try:
    got = norm(read(f"{out}/index.html"))
    expect(got == want_index, f"index.html differs.\n  want: {want_index}\n  got:  {got}")
except OSError as e:
    failures.append(f"index.html: {e}")

def snapshot():
    return {n: os.stat(os.path.join(out, n)).st_mtime_ns for n in os.listdir(out)} if os.path.isdir(out) else {}

time.sleep(1.1)
before = snapshot()
r = build(src, out)
expect(r.returncode == 0 and counts(r) == (0, 5, 0), f"unchanged rebuild summary: {r.stdout.strip().splitlines()[-1:]}")
expect(snapshot() == before, "an unchanged rebuild rewrote files")

time.sleep(1.1)
write(f"{src}/content/second.md", "---\ntitle: Second, edited\ndate: 2026-03-01\n---\nJust text, edited.\n")
before = snapshot()
r = build(src, out)
after = snapshot()
changed = sorted(n for n in after if after[n] != before.get(n))
expect(counts(r) == (2, 3, 0) and changed == ["index.html", "second.html"],
       f"one post edited: summary {r.stdout.strip().splitlines()[-1:]}, rewrote {changed}")
expect("Just text, edited." in read(f"{out}/second.html"), "second.html was not rebuilt with the new text")

time.sleep(1.1)
write(f"{src}/templates/post.html", read(f"{src}/templates/post.html").replace("<html>", "<html lang=\"en\">"))
before = snapshot()
r = build(src, out)
after = snapshot()
changed = sorted(n for n in after if after[n] != before.get(n))
expect(counts(r) == (4, 1, 0) and changed == ["alpha.html", "hello.html", "second.html", "undated.html"],
       f"post template edited: summary {r.stdout.strip().splitlines()[-1:]}, rewrote {changed}")

time.sleep(1.1)
os.remove(f"{src}/content/undated.md")
r = build(src, out)
expect(counts(r) == (1, 3, 1) and not os.path.exists(f"{out}/undated.html"),
       f"post deleted: summary {r.stdout.strip().splitlines()[-1:]}, undated.html exists={os.path.exists(f'{out}/undated.html')}")
expect("undated.html" not in read(f"{out}/index.html"), "index.html still links the deleted post")

# A fresh OUT from a changed SRC still gives full output (no stale state elsewhere).
r = build(src, os.path.join(tmp, "out2"))
expect(r.returncode == 0 and counts(r) == (4, 0, 0), f"build into a new OUT: {r.stdout.strip().splitlines()[-1:]}")

# Errors.
bad = os.path.join(tmp, "bad")
write(f"{bad}/templates/post.html", "{{ content }}")
write(f"{bad}/templates/index.html", "{{ posts }}")
write(f"{bad}/content/x.md", "---\ndate: 2026-01-01\n---\nno title\n")
r = build(bad, os.path.join(tmp, "out3"))
expect(r.returncode == 1 and r.stderr.strip(), f"post without a title: expected exit 1 + stderr, got {r.returncode}")
r = build(os.path.join(tmp, "nowhere"), os.path.join(tmp, "out4"))
expect(r.returncode == 1 and r.stderr.strip(), f"missing SRC/content: expected exit 1 + stderr, got {r.returncode}")

r = subprocess.run([sys.executable, "-m", "unittest"], capture_output=True, text=True, timeout=300)
expect(r.returncode == 0 and "Ran 0 tests" not in r.stderr, f"python3 -m unittest: exit {r.returncode}: {r.stderr[-400:]}")

print("\n".join(f"FAIL: {f}" for f in failures) or "PASS")
sys.exit(1 if failures else 0)
PY
