import os
import requests
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()


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

    def _make_request(self, method, endpoint, **kwargs):
        """
        A private helper method to make authenticated requests to the Zammad API.

        Args:
            method (str): The HTTP method (e.g., 'get', 'post', 'put').
            endpoint (str): The API endpoint (e.g., 'tickets', 'users/me').
            **kwargs: Additional keyword arguments passed to requests.request.

        Returns:
            dict or list: The JSON response from the API.

        Raises:
            requests.exceptions.RequestException: For connection errors or HTTP error status codes.
        """
        url = f"{self.api_url}/api/v1/{endpoint}"

        try:
            response = requests.request(method, url, headers=self.headers, timeout=15, **kwargs)
            # Raise an exception if the response was an HTTP error (4xx or 5xx)
            response.raise_for_status()
            # If response is successful but empty (e.g., 204 No Content), return None
            if response.status_code == 204:
                return None
            return response.json()
        except requests.exceptions.RequestException as e:
            # You would add more robust logging here in a real application
            print(f"API Request Error to {url}: {e}")
            raise

    # --- Methods Required by Your Tests ---

    def create_ticket(self, title: str, group: str, customer_id: int, article_body: str) -> dict:
        """
        Creates a new ticket in Zammad.

        Args:
            title (str): The title of the ticket.
            group (str): The name of the group to assign the ticket to.
            customer_id (int): The Zammad ID of the customer user.
            article_body (str): The initial message/body of the ticket.

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
        return self._make_request('post', 'tickets', json=payload)

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

    def get_self(self) -> dict:
        """
        Retrieves the user object associated with the API token.
        Useful for verifying authentication.

        Returns:
            dict: A dictionary representing the authenticated user.
        """
        return self._make_request('get', 'users/me')

    def create_user(self, email: str, firstname: str, lastname: str) -> dict:
        """
        Creates a new customer user.

        Args:
            email (str): The user's email address.
            firstname (str): The user's first name.
            lastname (str): The user's last name.

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
        return self._make_request('post', 'users', json=payload)

    def search_user(self, query: str) -> list:
        """
        Searches for users by a query string (e.g., email or name).

        Args:
            query (str): The search term.

        Returns:
            list: A list of user objects matching the query.
        """
        # The endpoint requires URL encoding for the query, but requests handles it.
        return self._make_request('get', f'users/search?query={query}')