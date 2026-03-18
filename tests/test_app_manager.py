# tests/test_app_manager.py

import pytest
from unittest.mock import patch, MagicMock
import sys

from src import app_manager


@patch('src.app_manager.Repo')
def test_update_app_success(mock_repo_class, monkeypatch):
    """Test update_app when git pull is successful and there are changes."""
    monkeypatch.setenv("REPO_PATH", "/fake/repo")

    mock_repo_instance = MagicMock()

    # Mock pull result with SUCCESS flag
    mock_pull_info = MagicMock()
    mock_pull_info.flags = 4  # Represents a successful pull with changes
    mock_repo_instance.remotes.origin.pull.return_value = [mock_pull_info]

    # Mock diff index to simulate file changes
    mock_diff_index = MagicMock()
    mock_diff_index.iter_change_type.return_value = [MagicMock(b_path="changed_file.py")]
    mock_repo_instance.index.diff.return_value = mock_diff_index

    mock_repo_class.return_value = mock_repo_instance

    result = app_manager.update_app()

    assert "Pull successful" in result
    mock_repo_instance.remotes.origin.pull.assert_called_once()


@patch('src.app_manager.Repo')
def test_update_app_failure(mock_repo_class, monkeypatch):
    """Test update_app when git pull fails."""
    monkeypatch.setenv("REPO_PATH", "/fake/repo")

    mock_repo_instance = MagicMock()

    # Mock pull result without SUCCESS flag
    mock_pull_info = MagicMock()
    mock_pull_info.flags = 0  # Represents a failed or no-change pull
    mock_repo_instance.remotes.origin.pull.return_value = [mock_pull_info]

    mock_repo_class.return_value = mock_repo_instance

    result = app_manager.update_app()

    assert result == "Pull failed."


def test_update_app_no_repo_path(monkeypatch):
    """Test update_app when REPO_PATH environment variable is not set."""
    monkeypatch.delenv("REPO_PATH", raising=False)
    result = app_manager.update_app()
    assert "REPO_PATH not configured" in result


@patch('src.app_manager.os.execl')
@patch('src.app_manager.os.close')
@patch('src.app_manager.psutil.Process')
@patch('src.app_manager.os.getpid', return_value=12345)
def test_restart_app(mock_getpid, mock_psutil_process, mock_os_close, mock_os_execl):
    """Test the application restart logic."""
    # Mock process to return dummy file handlers
    mock_process_instance = MagicMock()
    mock_handler1 = MagicMock(fd=1)
    mock_handler2 = MagicMock(fd=2)
    mock_process_instance.open_files.return_value = [mock_handler1]
    mock_process_instance.connections.return_value = [mock_handler2]
    mock_psutil_process.return_value = mock_process_instance

    # Mock sys arguments
    original_sys_argv = sys.argv
    sys.argv = ["/path/to/script.py"]

    app_manager.restart_app()

    mock_getpid.assert_called_once()
    mock_psutil_process.assert_called_once_with(12345)

    # Check that os.close was called for each file descriptor
    mock_os_close.assert_any_call(1)
    mock_os_close.assert_any_call(2)
    assert mock_os_close.call_count == 2

    # Check that os.execl was called correctly
    python_executable = sys.executable
    mock_os_execl.assert_called_once_with(python_executable, python_executable, '"{}"'.format(sys.argv[0]))

    # Restore sys.argv
    sys.argv = original_sys_argv


def test_stop_app():
    """Test the application stop logic."""
    with pytest.raises(SystemExit):
        app_manager.stop_app()
