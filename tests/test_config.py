# tests/test_config.py



import importlib
from pathlib import Path
import pytest
from config import global_config

def test_web_interface_defaults_to_false_without_env_var(monkeypatch):
    """Verify that WEB_INTERFACE defaults to False when the env var is absent."""
    # Mock dotenv.load_dotenv globally in sys.modules so it's not re-imported as the real function
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
    
    monkeypatch.delenv("WEB_INTERFACE", raising=False)
    
    # Reload the config module with the updated environment and mocked load_dotenv
    importlib.reload(global_config)
    
    assert global_config.WEB_INTERFACE is False

def test_web_interface_is_true_when_env_var_true(monkeypatch):
    """Verify that WEB_INTERFACE parses to True when set to 'true'."""
    monkeypatch.setenv("WEB_INTERFACE", "true")
    importlib.reload(global_config)
    assert global_config.WEB_INTERFACE is True

def test_web_interface_is_false_when_env_var_false(monkeypatch):
    """Verify that WEB_INTERFACE parses to False when set to 'false'."""
    monkeypatch.setenv("WEB_INTERFACE", "false")
    importlib.reload(global_config)
    assert global_config.WEB_INTERFACE is False

def test_local_tz_defaults_to_new_york_without_env_var(monkeypatch):
    """DP-252: LOCAL_TZ absent → America/New_York, and it resolves via tzdata."""
    import dotenv
    from zoneinfo import ZoneInfo
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("LOCAL_TZ", raising=False)
    importlib.reload(global_config)
    assert global_config.LOCAL_TZ == "America/New_York"
    ZoneInfo(global_config.LOCAL_TZ)  # raises if no tz database (Windows w/o tzdata)

def test_local_tz_from_env_var(monkeypatch):
    """DP-252: LOCAL_TZ present → used verbatim."""
    monkeypatch.setenv("LOCAL_TZ", "Pacific/Guam")
    importlib.reload(global_config)
    assert global_config.LOCAL_TZ == "Pacific/Guam"
    monkeypatch.delenv("LOCAL_TZ")
    importlib.reload(global_config)


@pytest.fixture
def load_config_outside_tests():
    """Returns a loader that reloads global_config as a non-test process would
    see it, with DATA_DIR set (or unset, for None). The testing override pins
    DATA_DIR to tests/test_data, which hides the value this is here to check;
    mkdir is stubbed so the reload creates nothing. Its own MonkeyPatch, so the
    environment is back before the closing reload restores the test config."""
    import dotenv
    from pathlib import Path

    with pytest.MonkeyPatch.context() as mp:
        def load(data_dir):
            mp.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
            mp.setattr(Path, "mkdir", lambda *args, **kwargs: None)
            for name in ("PYTEST_CURRENT_TEST", "APP_ENV", "MEMORY_DATABASE_FILE", "DATA_DIR"):
                mp.delenv(name, raising=False)
            if data_dir is not None:
                mp.setenv("DATA_DIR", str(data_dir))
            importlib.reload(global_config)
        yield load
    importlib.reload(global_config)


def test_data_dir_defaults_inside_the_project(load_config_outside_tests):
    """DP-410: DATA_DIR absent → <project>/data, as on every dev box."""
    load_config_outside_tests(None)
    assert global_config.DATA_DIR == global_config.PROJECT_ROOT / "data"


def test_data_dir_from_env_moves_every_store_and_workspace(load_config_outside_tests, tmp_path):
    """DP-410: the container sets DATA_DIR=/data. Stores and agent workspaces
    all follow it, so none is left behind under the app tree."""
    from pathlib import Path
    load_config_outside_tests(tmp_path / "data")

    data_dir = (tmp_path / "data").resolve()
    assert global_config.DATA_DIR == data_dir
    for setting in ("MEMORY_DATABASE_FILE", "PERSONA_SAVE_FILE", "CC_FIXR_CLONE_DIR", "CC_NOTES_DIR",
                    "AGY_WORKSPACES_DIR", "CC_WORKSPACES_DIR"):
        value = Path(getattr(global_config, setting))
        assert data_dir in value.parents, f"{setting} = {value} did not follow DATA_DIR"
        assert global_config.PROJECT_ROOT not in value.parents


def test_rate_limit_gemma_4_tpr_defaults_to_256000(monkeypatch):
    """RATE_LIMIT_GEMMA_4_TPR defaults to 256000 when the env var is absent."""
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("RATE_LIMIT_GEMMA_4_TPR", raising=False)
    monkeypatch.delenv("RATE_LIMIT_GEMMA_4_RPD", raising=False)
    try:
        importlib.reload(global_config)
        assert global_config.RATE_LIMIT_GEMMA_4_TPR == 256000
    finally:
        importlib.reload(global_config)


def test_rate_limit_gemma_4_tpr_from_env(monkeypatch):
    """RATE_LIMIT_GEMMA_4_TPR reads its value from the environment."""
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("RATE_LIMIT_GEMMA_4_TPR", "500000")
    try:
        importlib.reload(global_config)
        assert global_config.RATE_LIMIT_GEMMA_4_TPR == 500000
    finally:
        importlib.reload(global_config)


def test_rate_limit_gemma_4_rpd_does_not_bleed_into_tpr(monkeypatch):
    """DP-411: RATE_LIMIT_GEMMA_4_RPD must not bleed into RATE_LIMIT_GEMMA_4_TPR."""
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("RATE_LIMIT_GEMMA_4_TPR", raising=False)
    monkeypatch.setenv("RATE_LIMIT_GEMMA_4_RPD", "9999")
    try:
        importlib.reload(global_config)
        assert global_config.RATE_LIMIT_GEMMA_4_RPD == 9999
        assert global_config.RATE_LIMIT_GEMMA_4_TPR == 256000
    finally:
        importlib.reload(global_config)


def test_scripts_default_db_paths_follow_data_dir(load_config_outside_tests, tmp_path):
    """DP-411: DB maintenance scripts resolve their default DB from
    config.global_config.MEMORY_DATABASE_FILE when DATA_DIR is customized."""
    from scripts import repair_mangled_kobold_history, cleanup_vertex_links, reset_memory_summaries
    custom_data = (tmp_path / "custom_data").resolve()
    load_config_outside_tests(custom_data)
    expected_db_path = str(custom_data / "user_memory.db")

    importlib.reload(repair_mangled_kobold_history)
    importlib.reload(cleanup_vertex_links)
    importlib.reload(reset_memory_summaries)
    try:
        assert repair_mangled_kobold_history.build_parser().get_default("db") == expected_db_path
        assert cleanup_vertex_links.build_parser().get_default("db") == expected_db_path
        assert reset_memory_summaries.build_parser().get_default("db") == Path(expected_db_path)
        assert reset_memory_summaries.DEFAULT_DB == Path(expected_db_path)
    finally:
        load_config_outside_tests(None)
        importlib.reload(repair_mangled_kobold_history)
        importlib.reload(cleanup_vertex_links)
        importlib.reload(reset_memory_summaries)
