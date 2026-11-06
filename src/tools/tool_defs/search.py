"""Web/grounding search tools (no service_binding — core)."""

from typing import Any, Dict, List


SEARCH_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "google_grounding",
        "is_write": False,
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
            "name": "pubmed_search",
            "description": "Searches PubMed (NCBI) for biomedical literature. Returns PMID, "
                           "title, authors, journal, year, DOI, abstract, and a free PMC "
                           "full-text link when one exists. Accepts PubMed query syntax: "
                           "field tags like `crispr[tiab]`, `smith j[au]`, `2023:2025[dp]`, "
                           "`review[pt]`, and AND/OR/NOT.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "PubMed search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of articles to return (default: 5, "
                                       "capped by PUBMED_SEARCH_LIMIT_MAX).",
                        "default": 5,
                    },
                    "sort": {
                        "type": "string",
                        "enum": ["relevance", "date"],
                        "description": "`relevance` (default) or `date` (newest first).",
                        "default": "relevance",
                    },
                },
                "required": ["query"],
            },
        },
    },
]
