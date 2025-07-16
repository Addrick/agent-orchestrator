import os
import sys
import logging
from git import Repo
import psutil

logger = logging.getLogger(__name__)

"""
For use with remote development and redeployment
- update code from github
- restart the application with new changes applied
"""


def update_app():
    # Path to local repository
    repo_path = 'C:\\Users\\Adam\\Programming\\Python\\derpr-python'
    # 'https://github.com/Addrick/derpr-python/tree/master'

    # Open the repository
    repo = Repo(repo_path)

    # Pull the latest changes from the remote repository
    origin = repo.remotes.origin
    result = origin.pull()

    if (result[0].flags & 4) != 0:
        logging.warning("Pull successful. Changed files:")
        # Get the changes after the pull
        # changes = repo.git.diff('HEAD@{1}', 'HEAD')
        diff_index = repo.index.diff(None)
        for diff_added in diff_index.iter_change_type('A'):
            logging.warning(f"Added: {diff_added.b_path}")
        for diff_modified in diff_index.iter_change_type('M'):
            logging.warning(f"Modified: {diff_modified.b_path}")
        for diff_deleted in diff_index.iter_change_type('D'):
            logging.warning(f"Deleted: {diff_deleted.b_path}")
        return f"Pull successful, {diff_index.iter_change_type('M')} files modified."

    else:
        return "Pull failed."


def restart_app():
    """Restarts the current program, with file objects and descriptors
       cleanup
    """
    logger.info('Restarting application...')

    try:
        p = psutil.Process(os.getpid())
        for handler in p.open_files() + p.connections():
            if handler.fd:
                os.close(handler.fd)
    except Exception as e:
        logging.error(e)

    python = sys.executable
    os.execl(python, python, "\"{}\"".format(sys.argv[0]))


def stop_app():
    print("Stopping the program...")
    sys.exit()

# update_app()
# restart_app()
# restart_program()
