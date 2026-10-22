set -u
[ -f shop/cart.py ] || { echo "FAIL: shop/cart.py missing"; exit 1; }
python3 -I - <<'PY'
import os, subprocess, sys
from decimal import Decimal

# Hidden-check hardening (DP-449 review): `-I` keeps the task folder off this
# interpreter's path, and nothing at its top level may shadow the standard
# library for the child Pythons the check runs from it (`-m unittest` puts the
# working directory first on their path).
_shadows = sorted(n for n in os.listdir(".") if n.removesuffix(".py") in sys.stdlib_module_names)
if _shadows:
    print(f"FAIL: these names shadow Python's standard library: {', '.join(_shadows)}")
    sys.exit(1)

failures = []
r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."],
                   capture_output=True, text=True, timeout=120)
if r.returncode != 0:
    failures.append(f"the given tests fail: {r.stderr[-600:]}")

sys.path.insert(0, ".")
try:
    from shop.cart import Cart
except Exception as e:  # noqa: BLE001
    print(f"FAIL: cannot import shop.cart: {e!r}")
    sys.exit(1)

def line(qty, price="2.00"):
    c = Cart()
    c.add("x", price, qty)
    return c.line_total("x")

for qty, want in ((9, "18.00"), (10, "18.00"), (11, "19.80"), (1, "2.00")):
    if line(qty) != Decimal(want):
        failures.append(f"line_total for {qty} x 2.00: expected {want}, got {line(qty)}")

c = Cart()
c.add("a", "0.50", 4)
c.add("b", "1.25")
if c.total(0) != Decimal("3.25") or c.total(10) != Decimal("2.93"):
    failures.append(f"order total changed: total(0)={c.total(0)}, total(10)={c.total(10)}")
c = Cart()
c.add("a", "0.50", 2)
c.add("a", "9.99", 8)
if c.items["a"] != (Decimal("0.50"), 10) or c.line_total("a") != Decimal("4.50"):
    failures.append(f"merged line: items={c.items['a']}, line_total={c.line_total('a')}")
try:
    Cart().add("a", "1", 0)
    failures.append("add with qty 0 no longer raises")
except ValueError:
    pass

print("\n".join(f"FAIL: {f}" for f in failures) or "PASS")
sys.exit(1 if failures else 0)
PY
status=$?
# The given tests must be unchanged.
python3 -I - <<'PY' || status=1
import hashlib, sys
want = "849dc2ef4ae076247523ccff30909689ead60347f0bf31b4c060f6ea050cfd7a"
got = hashlib.sha256(open("tests/test_cart.py", "rb").read().replace(b"\r\n", b"\n")).hexdigest()
if got != want:
    print("FAIL: tests/test_cart.py was changed")
    sys.exit(1)
PY
exit $status
