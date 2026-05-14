from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
from pathlib import Path

from .results import write_run, diff_runs


def _load_suite(name: str):
    """Suites live at eval_harnesses.suites.<name> and must export
    `build_suite()` returning a SuiteSpec."""
    mod = importlib.import_module(f"eval_harnesses.suites.{name}")
    if not hasattr(mod, "build_suite"):
        raise SystemExit(f"suite '{name}' missing build_suite()")
    return mod.build_suite()


def _print_progress(cell):
    flag = "PASS" if cell.passed else "FAIL"
    print(f"  [{flag}] {cell.scenario_id} | mem={cell.memory_variant_id} prompt={cell.prompt_variant_id}")


async def _cmd_run(args):
    from .runner import run_suite
    spec = _load_suite(args.suite)
    print(f"Running suite '{spec.name}': {len(spec.scenarios)} scenarios x {len(spec.variants.cells())} variant cells (live={args.live})")
    run = await run_suite(
        spec,
        live=args.live,
        scenario_filter=args.scenarios,
        variant_filter=args.variants,
        progress=_print_progress,
    )
    out = write_run(run, args.results_dir)
    s = run.summary()
    print(f"\nDone. Passed {s['passed']}/{s['total']}. Wrote {out}")
    _print_variant_table(s.get("per_variant", {}))


def _print_variant_table(per_variant):
    if not per_variant:
        return
    rows = sorted(per_variant.items(), key=lambda kv: (-kv[1]["avg_score"], -kv[1]["passed"]))
    width = max((len(k) for k in per_variant), default=20)
    print(f"\n{'variant':<{width}}  pass/total  avg_score")
    print(f"{'-' * width}  ----------  ---------")
    for key, slot in rows:
        print(f"{key:<{width}}  {slot['passed']:>4}/{slot['total']:<5}  {slot['avg_score']:.3f}")


def _cmd_list(args):
    spec = _load_suite(args.suite)
    print(f"Suite: {spec.name}")
    print(f"Scenarios ({len(spec.scenarios)}):")
    for s in spec.scenarios:
        print(f"  - {s.id}: {s.description}")
    print(f"Memory variants ({len(spec.variants.memory)}):")
    for m in spec.variants.memory:
        print(f"  - {m.id}: {m.description}")
    print(f"Prompt variants ({len(spec.variants.prompt)}):")
    for p in spec.variants.prompt:
        print(f"  - {p.id}: {p.description}")


def _cmd_diff(args):
    print(diff_runs(args.a, args.b))


def _cmd_report(args):
    import json
    payload = json.loads(Path(args.run).read_text(encoding="utf-8"))
    s = payload.get("summary", {})
    print(f"Suite: {payload.get('suite')}")
    print(f"Run:   {payload.get('started_at')} -> {payload.get('finished_at')} (live={payload.get('live')})")
    print(f"Total: {s.get('passed')}/{s.get('total')}")
    print(f"\nPer-grader:")
    for g, counts in (s.get("per_grader") or {}).items():
        print(f"  {g}: pass={counts['pass']} fail={counts['fail']}")
    _print_variant_table(s.get("per_variant", {}))
    if args.cells:
        print("\nCells:")
        for c in payload.get("cells", []):
            flag = "PASS" if c["passed"] else "FAIL"
            notes = "; ".join(g.get("notes", "") for g in c.get("grades", []))
            print(f"  [{flag}] {c['scenario_id']} | mem={c['memory_variant_id']} | {notes}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="eval_harnesses.framework")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="execute a suite")
    pr.add_argument("--suite", required=True, help="suite module name under eval_harnesses.suites")
    pr.add_argument("--live", action="store_true", help="use live LLM/services instead of mocks")
    pr.add_argument("--scenarios", nargs="*", help="filter to these scenario ids")
    pr.add_argument("--variants", nargs="*", help="filter to these variant ids (memory or prompt)")
    pr.add_argument("--results-dir", default="eval_harnesses/results")
    pr.set_defaults(func=_cmd_run, is_async=True)

    pl = sub.add_parser("list", help="list scenarios + variants in a suite")
    pl.add_argument("--suite", required=True)
    pl.set_defaults(func=_cmd_list, is_async=False)

    pd = sub.add_parser("diff", help="compare two run JSONs")
    pd.add_argument("a")
    pd.add_argument("b")
    pd.set_defaults(func=_cmd_diff, is_async=False)

    prep = sub.add_parser("report", help="re-render a result JSON")
    prep.add_argument("run", help="path to run JSON file")
    prep.add_argument("--cells", action="store_true", help="also list every cell")
    prep.set_defaults(func=_cmd_report, is_async=False)

    args = p.parse_args(argv)
    if getattr(args, "is_async", False):
        asyncio.run(args.func(args))
    else:
        args.func(args)


if __name__ == "__main__":
    main()
