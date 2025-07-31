# src/clients/zammad_client.py

import os
import requests
import logging
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any

# Load environment variables from a .env file if it exists
load_dotenv()
logger = logging.getLogger(__name__)


class ZammadClient:
    """
    A client for interacting with the Zammad API.

    This class handles authentication and provides methods for common
    actions like creating users and tickets.
    """

    def __init__(self):
        """
        Initializes the ZammadClient.

        It retrieves the Zammad URL and API token from environment variables.

        Raises:
            ValueError: If ZAMMAD_URL or ZAMMAD_API_TOKEN are not set in the environment.
        """
        self.api_url = os.environ.get("ZAMMAD_URL")
        self.api_token = os.environ.get("ZAMMAD_API_KEY")

        if not self.api_url or not self.api_token:
            raise ValueError("ZAMMAD_URL and ZAMMAD_API_KEY must be set in .env")

        # Prepare the authentication header for all subsequent requests
        self.headers = {
            'Authorization': f'Token token={self.api_token}',
            'Content-Type': 'application/json'
        }

    def _make_request(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None, **kwargs):
        """
        A private helper method to make authenticated requests to the Zammad API.

        Args:
            method (str): The HTTP method (e.g., 'get', 'post', 'put').
            endpoint (str): The API endpoint (e.g., 'tickets', 'users/me').
            params (Optional[Dict[str, Any]]): A dictionary of URL parameters for the request.
            **kwargs: Additional keyword arguments passed to requests.request.

        Returns:
            dict or list: The JSON response from the API.

        Raises:
            requests.exceptions.RequestException: For connection errors or HTTP error status codes.
        """
        url = f"{self.api_url}/api/v1/{endpoint}"

        try:
            # Pass params directly to requests to handle URL encoding safely.
            response = requests.request(method, url, params=params, headers=self.headers, timeout=15, **kwargs)
            # Raise an exception if the response was an HTTP error (4xx or 5xx)
            response.raise_for_status()

            # If the response has no content, we cannot parse JSON.
            # This handles 204 No Content, and also 200 OK with an empty body from DELETE requests.
            if not response.content:
                return None

            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Zammad API Request Error to {method.upper()} {url}: {e}", exc_info=True)
            raise

    # --- Ticket Methods ---

    def create_ticket(self, title: str, group: str, customer_id: int, article_body: str, tags: Optional[List[str]] = None) -> dict:
        """
        Creates a new ticket in Zammad.

        Args:
            title (str): The title of the ticket.
            group (str): The name of the group to assign the ticket to.
            customer_id (int): The Zammad ID of the customer user.
            article_body (str): The initial message/body of the ticket.
            tags (Optional[List[str]]): A list of tags to add to the ticket.

        Returns:
            dict: A dictionary representing the newly created ticket.
        """
        payload = {
            "title": title,
            "group": group,
            "customer_id": customer_id,
            "article": {
                "body": article_body,
                "type": "note",
                "internal": False
            }
        }
        # Zammad's API expects tags as a comma-separated string
        if tags:
            payload['tags'] = ','.join(tags)
        return self._make_request('post', 'tickets', json=payload)

    def delete_ticket(self, ticket_id: int) -> None:
        """
        Deletes a ticket from Zammad.

        Args:
            ticket_id (int): The ID of the ticket to delete.
        """
        return self._make_request('delete', f'tickets/{ticket_id}')

    def add_article_to_ticket(self, ticket_id: int, body: str, internal: bool = False) -> dict:
        """
        Adds a new article (a message or note) to an existing ticket.

        Args:
            ticket_id (int): The ID of the ticket to add the article to.
            body (str): The content of the message.
            internal (bool): Whether the article is an internal note (default False).

        Returns:
            dict: A dictionary representing the newly created article.
        """
        payload = {
            "ticket_id": ticket_id,
            "body": body,
            "type": "note",
            "internal": internal
        }
        return self._make_request('post', 'ticket_articles', json=payload)

    def search_tickets(self, query: str, limit: int = 50) -> list:
        """
        Searches for tickets by a query string.

        Args:
            query (str): The search term (e.g., 'customer_id:123').
            limit (int): The maximum number of tickets to return.

        Returns:
            list: A list of ticket objects matching the query.
        """
        params = {'query': query, 'limit': limit}
        return self._make_request('get', 'tickets/search', params=params)

    # --- User Methods ---

    def get_self(self) -> dict:
        """
        Retrieves the user object associated with the API token.
        Useful for verifying authentication.

        Returns:
            dict: A dictionary representing the authenticated user.
        """
        return self._make_request('get', 'users/me')

    def create_user(self, email: str, firstname: str, lastname: str, note: Optional[str] = None) -> dict:
        """
        Creates a new customer user.

        Args:
            email (str): The user's email address.
            firstname (str): The user's first name.
            lastname (str): The user's last name.
            note (Optional[str]): An optional note for the user's profile.

        Returns:
            dict: A dictionary representing the newly created user.
        """
        payload = {
            "firstname": firstname,
            "lastname": lastname,
            "email": email,
            "roles": ["Customer"],  # Explicitly set role
            "active": True
        }
        if note:
            payload['note'] = note
        return self._make_request('post', 'users', json=payload)

    def delete_user(self, user_id: int) -> None:
        """
        Deletes a user from Zammad.

        Args:
            user_id (int): The ID of the user to delete.
        """
        return self._make_request('delete', f'users/{user_id}')

    def search_user(self, query: str) -> list:
        """
        Searches for users by a query string (e.g., email or name).

        Args:
            query (str): The search term.

        Returns:
            list: A list of user objects matching the query.
        """
        params = {'query': query}
        return self._make_request('get', 'users/search', params=params)
