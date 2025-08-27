# src/app_manager.py

import os
import sys
import logging
from git import Repo, Diff, IndexFile
import psutil
from typing import List, Union

logger = logging.getLogger(__name__)

"""
For use with remote development and redeployment
- update code from .github
- restart the application with new changes applied
"""


def update_app() -> str:
    # Path to local repository
    repo_path = os.environ.get("REPO_PATH")
    if not repo_path:
        logger.error("REPO_PATH environment variable not set. Cannot update app.")
        return "Update failed: REPO_PATH not configured."
    # 'https://github.com/Addrick/derpr-python/tree/master'

    # Open the repository
    repo = Repo(repo_path)

    # Pull the latest changes from the remote repository
    origin = repo.remotes.origin
    result = origin.pull()

    if (result[0].flags & 4) != 0:
        logger.warning("Pull successful. Changed files:")
        # Get the changes after the pull
        # changes = repo.git.diff('HEAD@{1}', 'HEAD')
        diff_index: IndexFile = repo.index
        diffs: List[Diff] = diff_index.diff(None)
        for diff_added in diffs.iter_change_type('A'):
            logger.warning(f"Added: {diff_added.b_path}")
        for diff_modified in diffs.iter_change_type('M'):
            logger.warning(f"Modified: {diff_modified.b_path}")
        for diff_deleted in diffs.iter_change_type('D'):
            logger.warning(f"Deleted: {diff_deleted.b_path}")

        modified_files_count = len(list(diffs.iter_change_type('M')))
        return f"Pull successful, {modified_files_count} files modified."

    else:
        return "Pull failed."


def restart_app() -> None:
    """Restarts the current program, with file objects and descriptors
       cleanup
    """
    logger.info('Restarting application...')

    try:
        p = psutil.Process(os.getpid())
        open_items: List[Union[psutil.Process, psutil.Connection]] = p.open_files() + p.connections()
        for handler in open_items:
            if hasattr(handler, 'fd') and handler.fd:
                os.close(handler.fd)
    except Exception as e:
        logger.error(e)

    python = sys.executable
    os.execl(python, python, "\"{}\"".format(sys.argv[0]))


def stop_app() -> None:
    print("Stopping the program...")
    sys.exit()
