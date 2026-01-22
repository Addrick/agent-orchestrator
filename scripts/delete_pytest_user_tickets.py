# scripts/cleanup_zammad_tests.py

import os
import sys
import logging

# Ensure the project root is in the Python path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.clients.zammad_client import ZammadClient
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TEST_USER_EMAIL = "pytest-lifecycle-user@zammad.local"
KEYWORDS = ["FluxCapacitor", "WarpDrive", "[Test]", "[Pytest Automated Test Ticket]"]


def cleanup():
    load_dotenv()

    try:
        client = ZammadClient()
        logger.info(f"Connected to Zammad at {client.api_url}")
    except Exception as e:
        logger.error(f"Failed to connect to Zammad: {e}")
        return

    # 1. Find the Test User
    logger.info(f"Searching for test user: {TEST_USER_EMAIL}...")
    users = client.search_user(f"email:{TEST_USER_EMAIL}")

    user_id = None
    if users:
        user_id = users[0]['id']
        logger.info(f"Found Test User ID: {user_id}")

        # 2. Delete all tickets for this user
        logger.info("Fetching tickets for test user...")
        # Pagination might be needed if you have > 50, but let's start with default limit
        tickets = client.search_tickets(f"customer_id:{user_id}", limit=200)

        if tickets:
            logger.info(f"Found {len(tickets)} tickets belonging to user {user_id}. Deleting...")
            for t in tickets:
                try:
                    client.delete_ticket(t['id'])
                    logger.info(f"Deleted Ticket #{t['id']}: {t.get('title', 'No Title')}")
                except Exception as e:
                    logger.error(f"Failed to delete ticket #{t['id']}: {e}")
        else:
            logger.info("No tickets found for this user.")
    else:
        logger.info("Test user not found.")

    # 3. Cleanup Orphaned Tickets by Title (Just in case)
    logger.info("Scanning for orphaned test tickets by keyword...")
    for keyword in KEYWORDS:
        query = f'title:"{keyword}"'
        tickets = client.search_tickets(query, limit=50)
        for t in tickets:
            try:
                client.delete_ticket(t['id'])
                logger.info(f"Deleted Orphaned Ticket #{t['id']}: {t.get('title', 'No Title')}")
            except Exception as e:
                # 404 means it was probably already deleted in step 2
                if "404" not in str(e):
                    logger.error(f"Failed to delete orphan #{t['id']}: {e}")

    # 4. Delete the User
    if user_id:
        logger.info(f"Deleting Test User ID: {user_id}...")
        try:
            client.delete_user(user_id)
            logger.info("User deleted successfully.")
        except Exception as e:
            logger.error(f"Failed to delete user: {e}")

    logger.info("Cleanup complete.")


if __name__ == "__main__":
    cleanup()