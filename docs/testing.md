# Testing

Test conventions for this repo. `CLAUDE.md` points here; it does not restate any of it.

## Commands

```bash
pytest -m "not llm_live" -n auto     # default: parallel, skips live LLM calls
pytest                               # everything (live tiers auto-skip without creds)
pytest -m "not zammad_live and not llm_live and not discord_live" -n auto   # unit + integration
pytest -m "not integration and not zammad_live and not llm_live and not discord_live" -n auto  # unit only
pytest -m "zammad_live"              # Zammad live only
pytest -m "llm_live"                 # LLM live only
pytest tests/test_engine.py          # single file
pytest --cov=src                     # with coverage
```

## The four tiers

Ordered by execution cost:

1. **Unit** (no marker) — one component, everything mocked, no network.
2. **Integration** (`@pytest.mark.integration`) — multi-component flows with mocked
   externals, no network.
3. **Zammad live** (`@pytest.mark.zammad_live`) — needs a live Zammad
   (`ZAMMAD_URL` + `ZAMMAD_API_KEY`).
4. **LLM live** (`@pytest.mark.llm_live`) — real provider API calls, needs API keys.

Live tiers auto-skip when credentials are absent (`tests/conftest.py`). Test Zammad
credentials live in `.env.test` (gitignored), loaded with `override=True` so a test run
can never reach production. Committed fixture files: `tests/fixtures/`. ⚠️ `tests/test_data/`
is **gitignored** — it is the test run's scratch `DATA_DIR` (databases, logs), so a fixture
put there passes locally and is missing in CI.

## Mandatory test requirements

Changing any of the following **requires** matching tests in the same commit.

### Database schema (`memory_manager.py` CREATE TABLE / ALTER TABLE)

Add migration tests using the `legacy_mem_manager` fixture pattern in
`tests/memory/test_memory_manager.py`. The fixture builds a DB on the **old** schema,
then the test calls `create_schema()` and asserts the migration.

Must cover: column/table added · existing data preserved · indexes created · the new
feature usable on the migrated DB · idempotent on a second run.

> ⚠️ Unit tests against `:memory:` always start fresh and **cannot** catch migration
> bugs against a production database that already has rows.

> ⚠️ An index declared beside its `CREATE TABLE` in `schema_sql` is built **before** the
> ALTER that adds its column, so it takes boot down on exactly the deployments that
> already have data. Declare it with the ALTER, not the CREATE.

### Config schema (`agents.json`, `system_personas.json`, `default_personas.json`, `global_config.py`)

- Adding/renaming/removing a key: test both the key **absent** (old config files on
  disk) and **present**.
- A config value that drives runtime behavior (e.g. `notification_defaults.channel`)
  gets tested against realistic config, not just mocks.
- Agent config: via `AgentManager` dependency injection in
  `tests/agents/test_agent_manager.py`.
- Persona config: loading and field access in `tests/test_persona.py`.

> ⚠️ Merging a persona change does **not** deploy it. `config/optional_personas/` is a
> template; the live persona is hand-maintained in `data/personas.json` on a docker
> volume. A green test says nothing about prod.

### Cross-module contracts (imports, base-class APIs, interface signatures)

- Renaming or moving a class/function: grep every importer and update them in the
  **same** commit.
- Changing a base-class API (e.g. `Agent`, `AgentLoop`): update all subclasses and their
  tests in the same commit.
- Run `mypy src/ services/ --config-file mypy.ini` before committing any structural
  change.

### Startup registration (new `ServiceIntegration`, tool handler, or notifier)

If a component must be registered at startup to work at all, test that the registration
**happens** — not merely that the component works in isolation.
`tests/integration/test_startup_wiring.py` asserts every tool `service_binding` in
`ALL_TOOL_DEFINITIONS` has a registered handler; extend it when adding a service.

## Gates

`pytest`, `flake8 src/ services/`, and `mypy src/ services/ --config-file mypy.ini` —
the same three `.github/workflows/deploy.yml` runs. Nothing is `QA_READY` until all
three pass **inside the ticket's own worktree**, using that worktree's `.venv`.

> ⚠️ Run `pytest` from **inside** the worktree. `pythonpath = src .` resolves relative to
> the run directory, so `pytest worktrees/DP-XXX/...` from the main tree imports `src`
> from the *main* tree and silently masks real pass/fail.
