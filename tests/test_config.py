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
