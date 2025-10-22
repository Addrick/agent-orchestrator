# src/clients/zammad_client.py

import os
import requests
import logging
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from requests.exceptions import RequestException

# Load environment variables from a .env file if it exists
load_dotenv()
logger = logging.getLogger(__name__)


class ZammadClient:
    """
    A client for interacting with the Zammad API.

    This class handles authentication and provides methods for common
    actions like creating users and tickets.
    """

    def __init__(self) -> None:
        """
        Initializes the ZammadClient.

        It retrieves the Zammad URL and API token from environment variables.

        Raises:
            ValueError: If ZAMMAD_URL or ZAMMAD_API_TOKEN are not set in the environment.
        """
        self.api_url: Optional[str] = os.environ.get("ZAMMAD_URL")
        self.api_token: Optional[str] = os.environ.get("ZAMMAD_API_KEY")

        if not self.api_url or not self.api_token:
            raise ValueError("ZAMMAD_URL and ZAMMAD_API_KEY must be set in .env")

        # Prepare the authentication header for all subsequent requests
        self.base_headers: Dict[str, str] = {
            'Authorization': f'Token token={self.api_token}',
            'Content-Type': 'application/json'
        }

    def _make_request(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None,
                      impersonate_email: Optional[str] = None, **kwargs: Any) -> Any:
        """
        A private helper method to make authenticated requests to the Zammad API.
        """
        url = f"{self.api_url}/api/v1/{endpoint}"

        headers = self.base_headers.copy()
        if impersonate_email:
            headers['X-On-Behalf-Of'] = impersonate_email

        try:
            response = requests.request(method, url, params=params, headers=headers, timeout=15, **kwargs)
            response.raise_for_status()

            if not response.content:
                return None

            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Zammad API Request Error to {method.upper()} {url}: {e}", exc_info=True)
            raise

    # --- Ticket Methods ---

    def get_ticket(self, ticket_id: int) -> Dict[str, Any]:
        """
        Retrieves a single ticket by its ID, including all articles and related objects.
        """
        return self._make_request('get', f'tickets/{ticket_id}?expand=true')

    def create_ticket(self, title: str, group: str, customer_id: int, article_body: Optional[str] = None,
                      tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Creates a new ticket in Zammad. If article_body is omitted, creates an empty ticket.
        """
        payload: Dict[str, Any] = {
            "title": title,
            "group": group,
            "customer_id": customer_id,
        }
        if article_body:
            payload["article"] = {
                "body": article_body,
                "type": "note",
                "internal": False
            }
        if tags:
            payload['tags'] = ','.join(tags)
        return self._make_request('post', 'tickets', json=payload)

    def delete_ticket(self, ticket_id: int) -> None:
        """
        Deletes a ticket from Zammad.
        """
        self._make_request('delete', f'tickets/{ticket_id}')

    def add_article_to_ticket(self, ticket_id: int, body: str, internal: bool = False,
                              impersonate_email: Optional[str] = None) -> Dict[str, Any]:
        """
        Adds a new article (a message or note) to an existing ticket.
        """
        payload: Dict[str, Any] = {
            "ticket_id": ticket_id,
            "body": body,
            "type": "note",
            "internal": internal
        }
        return self._make_request('post', 'ticket_articles', json=payload, impersonate_email=impersonate_email)

    def update_ticket(self, ticket_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updates an existing ticket with the given payload.
        Example payload: {'state': 'closed'}
        """
        return self._make_request('put', f'tickets/{ticket_id}', json=payload)

    def search_tickets(self, query: str, limit: int = 50, sort_by: Optional[str] = None,
                       order_by: Optional[str] = 'desc') -> List[Dict[str, Any]]:
        """
        Searches for tickets by a query string with optional sorting.
        """
        params: Dict[str, Any] = {'query': query, 'limit': limit}
        if sort_by:
            params['sort_by'] = sort_by
            params['order_by'] = order_by
        return self._make_request('get', 'tickets/search', params=params)

    # --- User Methods ---

    def get_self(self) -> Dict[str, Any]:
        """
        Retrieves the user object associated with the API token.
        """
        return self._make_request('get', 'users/me')

    def create_user(self, email: str, firstname: str, lastname: str, note: Optional[str] = None) -> Dict[str, Any]:
        """
        Creates a new customer user.
        """
        payload: Dict[str, Any] = {
            "firstname": firstname,
            "lastname": lastname,
            "email": email,
            "roles": ["Customer"],
            "active": True
        }
        if note:
            payload['note'] = note
        return self._make_request('post', 'users', json=payload)

    def update_user(self, user_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updates an existing user with the given payload.
        """
        return self._make_request('put', f'users/{user_id}', json=payload)

    def delete_user(self, user_id: int) -> None:
        """
        Deletes a user from Zammad.
        """
        self._make_request('delete', f'users/{user_id}')

    def search_user(self, query: str) -> List[Dict[str, Any]]:
        """
        Searches for users by a query string.
        """
        params = {'query': query}
        return self._make_request('get', 'users/search', params=params)
