# scripts/cleanup_zammad.py

import os
import sys
import time

# This allows the script to import modules from the 'src' directory
# by adding the parent directory of 'scripts' to the Python path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.clients.zammad_client import ZammadClient

# --- CONFIGURATION ---
# Define the patterns to search for in the old test data.
TICKET_TITLE_PREFIX = "Integration Test Ticket"
USER_FIRSTNAME = "Integration"
# --- END CONFIGURATION ---


def main():
    """
    Finds and deletes test data from Zammad based on the configured patterns.
    This now uses a multi-stage process to handle object references.
    """
    print("--- Zammad Test Data Cleanup Utility ---")

    try:
        client = ZammadClient()
        print("Successfully connected to Zammad API.")
    except Exception as e:
        print(f"Error: Could not initialize Zammad client. Please check your .env file. Details: {e}")
        return

    # --- 1. Fetch all data ---
    print("\nFetching all users and tickets... (This may take a moment)")
    try:
        all_tickets = client.get_all_tickets()
        all_users = client.get_all_users()
        print(f"Found {len(all_tickets)} total tickets and {len(all_users)} total users.")
    except Exception as e:
        print(f"Error: Failed to fetch data from Zammad. Details: {e}")
        return

    # --- 2. Filter data for deletion ---
    tickets_to_delete = [
        t for t in all_tickets if t.get('title', '').startswith(TICKET_TITLE_PREFIX)
    ]
    users_to_delete = [
        u for u in all_users if u.get('firstname') == USER_FIRSTNAME
    ]

    # --- 3. Dry Run and Confirmation ---
    print("\n--- DRY RUN: The following items will be processed ---")

    if tickets_to_delete:
        print(f"\n[+] Found {len(tickets_to_delete)} test tickets to delete:")
        for ticket in tickets_to_delete:
            print(f"  - Ticket ID: {ticket['id']}, Title: \"{ticket['title']}\"")
    else:
        print("\n[+] No test tickets found to delete.")

    if users_to_delete:
        print(f"\n[+] Found {len(users_to_delete)} test users to deactivate and then delete:")
        for user in users_to_delete:
            print(f"  - User ID: {user['id']}, Name: {user['firstname']} {user['lastname']}, Email: {user['email']}")
    else:
        print("\n[+] No test users found to process.")

    if not tickets_to_delete and not users_to_delete:
        print("\nCleanup not required. Exiting.")
        return

    print("\n" + "="*50)
    print("WARNING: This is a destructive operation and cannot be undone.")
    print("Please review the list of items above carefully.")
    print("="*50)

    try:
        confirm = input("Type 'yes' to proceed with deletion: ")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return

    if confirm.lower() != 'yes':
        print("\nConfirmation not received. Aborting cleanup.")
        return

    # --- 4. Multi-Stage Deletion ---
    print("\n--- STAGE 1: DELETING TICKETS ---")
    for ticket in tickets_to_delete:
        try:
            print(f"Deleting ticket ID: {ticket['id']}...")
            client.delete_ticket(ticket['id'])
        except Exception as e:
            print(f"  - ERROR deleting ticket {ticket['id']}: {e}")

    # Proceed with user cleanup only if there are users to clean.
    if users_to_delete:
        print("\n--- STAGE 2: DEACTIVATING USERS ---")
        for user in users_to_delete:
            try:
                print(f"Deactivating user ID: {user['id']}...")
                client.update_user(user['id'], {"active": False})
            except Exception as e:
                print(f"  - ERROR deactivating user {user['id']}: {e}")

        print("\n--- STAGE 3: DELETING DEACTIVATED USERS ---")
        for user in users_to_delete:
            try:
                print(f"Deleting user ID: {user['id']}...")
                client.delete_user(user['id'])
            except Exception as e:
                print(f"  - ERROR deleting user {user['id']}: {e}")

    print("\n--- Cleanup complete. ---")


if __name__ == "__main__":
    main()
