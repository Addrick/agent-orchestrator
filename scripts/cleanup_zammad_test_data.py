"""
scripts/cleanup_zammad_test_data.py

Removes all test data left behind by the Zammad integration test suite.

Targets
-------
- All tickets belonging to the persistent test user (pytest-lifecycle-user@zammad.local)
- Any orphaned [Test] tickets not owned by that user (edge cases / partial runs)
- Optionally the test user account itself (--delete-user)

The bot agent user (autotriage@bot.local) is intentionally left alone; it is
a real operational account shared with production.

Usage
-----
  python scripts/cleanup_zammad_test_data.py                      # dry run
  python scripts/cleanup_zammad_test_data.py --force              # delete tickets
  python scripts/cleanup_zammad_test_data.py --force --delete-user
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from src.clients.zammad_client import ZammadClient

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
logger = logging.getLogger(__name__)

TEST_USER_EMAIL = "pytest-lifecycle-user@zammad.local"


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean up Zammad integration test data.")
    parser.add_argument(
        '--force', action='store_true',
        help='Actually perform deletions. Without this flag the script is a dry run.'
    )
    parser.add_argument(
        '--delete-user', action='store_true',
        help='Also delete the persistent test user account.'
    )
    args = parser.parse_args()

    load_dotenv()

    try:
        client = ZammadClient()
    except Exception as e:
        logger.error(f"Failed to connect to Zammad: {e}")
        sys.exit(1)

    dry_run = not args.force
    if dry_run:
        logger.info("DRY RUN — pass --force to actually delete.\n")

    # --- 1. Locate the persistent test user ---
    users = client.search_user(query=TEST_USER_EMAIL)
    user_id = users[0]['id'] if users else None

    if user_id:
        logger.info(f"Found test user  ID={user_id}  ({TEST_USER_EMAIL})")
    else:
        logger.info(f"Test user '{TEST_USER_EMAIL}' not found.")

    # --- 2. Collect tickets owned by the test user ---
    tickets_to_delete = []
    seen_ids: set = set()

    if user_id:
        user_tickets = client.search_tickets(query=f"customer_id:{user_id}", limit=200)
        for t in user_tickets:
            tickets_to_delete.append(t)
            seen_ids.add(t['id'])

    # --- 3. Collect orphaned [Test] tickets (different owner or partial runs) ---
    orphans = client.search_tickets(query='title:"[Test]"', limit=100)
    for t in orphans:
        if t['id'] not in seen_ids:
            tickets_to_delete.append(t)
            seen_ids.add(t['id'])

    # --- 4. Report ---
    if tickets_to_delete:
        logger.info(f"\nTickets found ({len(tickets_to_delete)}):")
        for t in tickets_to_delete:
            logger.info(f"  #{t['id']:>6}  {t['title']}")
    else:
        logger.info("No test tickets found.")

    # --- 5. Delete tickets ---
    if tickets_to_delete and not dry_run:
        logger.info("\nDeleting tickets...")
        for t in tickets_to_delete:
            try:
                client.delete_ticket(t['id'])
                logger.info(f"  Deleted #{t['id']}")
            except Exception as e:
                logger.error(f"  Failed to delete #{t['id']}: {e}")

    # --- 6. Optionally delete the test user ---
    if args.delete_user:
        if user_id:
            logger.info(f"\nTest user ID={user_id} marked for deletion.")
            if not dry_run:
                try:
                    client.delete_user(user_id)
                    logger.info("  Test user deleted.")
                except Exception as e:
                    logger.error(f"  Failed to delete test user: {e}")
        else:
            logger.info("\n--delete-user specified but test user was not found.")

    if dry_run:
        logger.info("\nDry run complete. Run with --force to apply.")


if __name__ == "__main__":
    main()
