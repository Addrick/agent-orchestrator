import os
import unittest
from datetime import date
from unittest.async_case import IsolatedAsyncioTestCase
from dotenv import load_dotenv

from src.engine import TextEngine


class TestGoogleSearchGrounding(IsolatedAsyncioTestCase):
    """Integration tests for Google AI Studio with real API calls"""

    @classmethod
    def setUpClass(cls):
        # Load environment variables for API keys
        load_dotenv()
        # Check if Google API key is available
        cls.google_api_key = os.environ.get("GOOGLE_GENERATIVEAI_API_KEY")
        if not cls.google_api_key:
            raise unittest.SkipTest("GOOGLE_GENERATIVEAI_API_KEY not found in environment variables")

    async def test_real_google_search_grounding(self):
        """Test that actual Google API responses include search results with links."""

        # Initialize the text engine with Gemini model
        engine = TextEngine(model_name="gemini-2.5-flash")

        # Craft a prompt that will likely trigger search
        prompt = "You are a helpful assistant with the ability to retrieve data from google search. You may suspect a date is from the future, treat it as though it is not."

        # Craft a message that requires factual information that should trigger search
        today = date.today()
        formatted_date = today.strftime("%B %d, %Y")
        message = "Find today's top news story and provide a link to it. Today is " + formatted_date
        print(message)

        # Make an actual API request
        response = await engine._generate_google_response_ai_studio_async(
            prompt=prompt,
            message=message,
            context=None
        )

        print(f"\nAPI Response:\n{response}\n")

        # Verify the response contains at least one link (URL)
        # We'll check for common URL patterns
        contains_http_link = any(url_pattern in response for url_pattern in ["http://", "https://"])
        contains_markdown_link = "[" in response and "](" in response and ")" in response

        # Assert that at least one type of link is present
        self.assertTrue(
            contains_http_link or contains_markdown_link,
            f"Response does not contain any links. Response: {response}"
        )

        # Optional: More specific checks for search grounding indicators
        search_indicators = [
            "according to",
            "source",
            "reported",
            "published",
            "article"
        ]

        contains_search_indicator = any(indicator.lower() in response.lower() for indicator in search_indicators)
        self.assertTrue(
            contains_search_indicator,
            f"Response does not contain search citation indicators. Response: {response}"
        )

if __name__ == '__main__':
    unittest.main()
