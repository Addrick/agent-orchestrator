"""Web/grounding search tools (no service_binding — core)."""

from typing import Any, Dict, List


SEARCH_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "google_grounding",
        "capabilities": {
            "produces_untrusted": True,
            "irreversible": False,
            "locality": "network",
            "sensitivity": "public",
        },
        "function": {
            "name": "google_grounding_search",
            "description": "Enables Google's native Search grounding feature for Gemini models. "
                           "Allows the model to retrieve up-to-date information from the web to "
                           "support its responses. Has no effect on non-Gemini models. "
                           "Subject to Google grounding API costs and rate limits.",
        },
    },
    {
        "type": "function",
        "is_write": False,
        "capabilities": {
            "produces_untrusted": True,
            "irreversible": False,
            "locality": "network",
            "sensitivity": "public",
        },
        "function": {
            "name": "web_search",
            "description": "Searches the web for information using DuckDuckGo. Returns titles, "
                           "URLs, and summaries for the most relevant results. Compatible with "
                           "all model providers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
]
