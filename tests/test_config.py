# tests/test_config.py

import os
import sys
import importlib
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
