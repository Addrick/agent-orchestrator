# Eval framework

Generic harness for testing how memory configuration and prompt variants
affect chatbot behavior. Memory-only scope today (zammad_merge stays
standalone).

## Concepts

- **Scenario** (`scenarios.py`) — one test case: seeded memory state +
  user request + expectations. Loaded from suite's `scenarios.json`.
- **Variants** (`variants.py`) — two axes:
  - `MemoryVariant`: which backends, retrieval params, seed config
  - `PromptVariant`: persona prompt body, system addendum
- **VariantMatrix** = cartesian product of memory × prompt variants.
- **Driver** (suite-specific) — function that exercises ChatSystem and
  produces a `RunOutput`. Recall suites call `retrieve_relevant_summaries`;
  behavioral suites call `generate_response`.
- **Grader** (`grading.py`) — scores a `RunOutput` against a scenario.
  Stubs ship: `contains`, `retrieval_hits`, `llm_judge` (stub).
- **Runner** (`runner.py`) — iterates scenarios × variant cells, builds
  fixture, calls driver, applies graders, collects `CellResult`s.
- **Results** (`results.py`) — JSON serialization + run diff.

## Layout

```
eval_harnesses/
  framework/         # this package — generic infra
  suites/
    memory_recall/   # first concrete suite (stub)
  results/           # timestamped run JSONs
```

## Adding a suite

```python
# eval_harnesses/suites/<name>/__init__.py
def build_suite() -> SuiteSpec:
    return SuiteSpec(
        name="<name>",
        scenarios=load_scenarios(<path>),
        variants=load_variants(<path>),
        driver=my_driver,             # async fn -> RunOutput
        default_graders=["contains"],
    )
```

`build_suite()` is the only required entry point; `cli.py` discovers it
via `importlib.import_module("eval_harnesses.suites.<name>")`.

## CLI

```powershell
$env:PYTHONPATH="."
python -m eval_harnesses.framework.cli list --suite memory_recall
python -m eval_harnesses.framework.cli run  --suite memory_recall [--live] [--scenarios id1 id2] [--variants v1 v2]
python -m eval_harnesses.framework.cli diff results/a.json results/b.json
```

## Known stubs (intentional)

- **MockLLM wiring** (`fixtures.py`) — `MockLLM` exists but is not yet
  patched into TextEngine. Each suite picks its patch point.
- **Seed memory insertion** (`fixtures._seed_memory`) — placeholder.
  Decide per suite whether to use raw turns + real summarizer or
  pre-built summaries.
- **Hindsight branch** in recall driver — no client call yet.
- **LLMJudgeGrader** — returns `passed=False, notes="not implemented"`.
- **Embedding step** in recall driver — `query_embeddings=None` today.

These are scaffolding holes, not bugs. Wire as suites need them.
