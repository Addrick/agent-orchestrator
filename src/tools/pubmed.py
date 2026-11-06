"""PubMed search via NCBI E-utilities (DP-501).

Two reads per call: ``esearch`` (query -> ranked PMIDs + total count) then
``efetch`` (PMIDs -> article XML with abstracts). ``esummary`` is not used
because it carries no abstract, and the abstract is the point of the tool.

Every hit is third-party text, so the page is clamped to
``PUBMED_SEARCH_LIMIT_MAX`` (an unparseable ``max_results`` included) and each
abstract to ``PUBMED_ABSTRACT_MAX_CHARS``. Cuts are flagged in the hit rather
than silent, per the "bound untrusted third-party text" row of
docs/capability_map.md.
"""

import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import httpx

from config import global_config
from src.security.vault import get_vault

logger = logging.getLogger(__name__)

_DEFAULT_RESULTS = 5
_MAX_AUTHORS = 6
_SORTS = {"relevance": "relevance", "date": "pub_date"}


def _err(message: str) -> Dict[str, Any]:
    return {"status": "error", "error": message}


def _text(el: Optional[ET.Element]) -> str:
    """All text under ``el``, inline markup (<i>, <sup>, ...) flattened."""
    if el is None:
        return ""
    return " ".join("".join(el.itertext()).split())


def _year(article: ET.Element) -> str:
    pub = article.find("Journal/JournalIssue/PubDate")
    if pub is None:
        return ""
    year = pub.findtext("Year")
    if year:
        return year.strip()
    return (pub.findtext("MedlineDate") or "").strip()[:4]


def _authors(article: ET.Element) -> List[str]:
    names = []
    for au in article.findall("AuthorList/Author"):
        collective = au.findtext("CollectiveName")
        if collective:
            names.append(collective.strip())
            continue
        last = (au.findtext("LastName") or "").strip()
        initials = (au.findtext("Initials") or "").strip()
        if last:
            names.append(f"{last} {initials}".strip())
    return names


def _abstract(article: ET.Element) -> str:
    parts = []
    for node in article.findall("Abstract/AbstractText"):
        body = _text(node)
        if not body:
            continue
        label = node.get("Label")
        parts.append(f"{label}: {body}" if label else body)
    return "\n".join(parts)


def parse_efetch(xml_text: str, max_chars: int) -> Dict[str, Dict[str, Any]]:
    """PubMed efetch XML -> ``{pmid: hit}``. Book records are skipped."""
    root = ET.fromstring(xml_text)
    hits: Dict[str, Dict[str, Any]] = {}
    for rec in root.findall("PubmedArticle"):
        citation = rec.find("MedlineCitation")
        if citation is None:
            continue
        pmid = (citation.findtext("PMID") or "").strip()
        article = citation.find("Article")
        if not pmid or article is None:
            continue
        ids = {
            a.get("IdType"): (a.text or "").strip()
            for a in rec.findall("PubmedData/ArticleIdList/ArticleId")
        }
        authors = _authors(article)
        abstract = _abstract(article)
        hit: Dict[str, Any] = {
            "pmid": pmid,
            "title": _text(article.find("ArticleTitle")),
            "authors": authors[:_MAX_AUTHORS],
            "journal": _text(article.find("Journal/Title")),
            "year": _year(article),
            "doi": ids.get("doi") or None,
            "pmcid": ids.get("pmc") or None,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "abstract": abstract[:max_chars] if abstract else None,
        }
        if len(authors) > _MAX_AUTHORS:
            hit["author_count"] = len(authors)
        if len(abstract) > max_chars:
            hit["abstract_truncated"] = True
        if hit["pmcid"]:
            hit["full_text_url"] = f"https://pmc.ncbi.nlm.nih.gov/articles/{hit['pmcid']}/"
        hits[pmid] = hit
    return hits


class PubMedSearchHandler:
    """Registers ``pubmed_search`` with a ToolManager."""

    def __init__(self, transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        # Injectable so tests drive it with httpx.MockTransport.
        self._transport = transport

    def register(self, manager: Any) -> None:
        manager.register("pubmed_search", self._pubmed_search)

    def _base_params(self) -> Dict[str, str]:
        params = {"db": "pubmed", "tool": "derpr"}
        if global_config.NCBI_EMAIL:
            params["email"] = global_config.NCBI_EMAIL
        api_key = get_vault().get("NCBI_API_KEY")
        if api_key:
            params["api_key"] = api_key
        return params

    async def _pubmed_search(
        self, query: str, max_results: int = _DEFAULT_RESULTS, sort: str = "relevance",
    ) -> Dict[str, Any]:
        ceiling = global_config.PUBMED_SEARCH_LIMIT_MAX
        try:
            capped = max(1, min(int(max_results), ceiling))
        except (TypeError, ValueError):
            # The fallback goes through the ceiling too (see hf_search).
            capped = max(1, min(_DEFAULT_RESULTS, ceiling))
        sort_key = _SORTS.get(str(sort).strip().lower(), "relevance")
        logger.info("Tool pubmed_search: %r max_results=%s sort=%s", query, capped, sort_key)
        if not str(query or "").strip():
            return _err("query is empty")

        base = global_config.PUBMED_API_BASE.rstrip("/")
        params = self._base_params()
        try:
            async with httpx.AsyncClient(
                timeout=global_config.PUBMED_HTTP_TIMEOUT, transport=self._transport,
            ) as client:
                r = await client.get(f"{base}/esearch.fcgi", params={
                    **params, "term": query, "retmax": str(capped),
                    "retmode": "json", "sort": sort_key,
                })
                r.raise_for_status()
                result = r.json().get("esearchresult", {})
                if "ERROR" in result:
                    return _err(f"PubMed rejected the query: {result['ERROR']}")
                pmids = [str(p) for p in result.get("idlist", [])]
                total = int(result.get("count", 0) or 0)
                payload: Dict[str, Any] = {
                    "status": "ok",
                    "query": query,
                    "query_translation": result.get("querytranslation"),
                    "total_matches": total,
                    "articles": [],
                }
                if not pmids:
                    payload["note"] = (
                        "No matches. PubMed ANDs every term, so drop terms or use "
                        "OR between synonyms; check query_translation for how the "
                        "terms were mapped."
                    )
                    return payload
                r = await client.get(f"{base}/efetch.fcgi", params={
                    **params, "id": ",".join(pmids),
                    "rettype": "abstract", "retmode": "xml",
                })
                r.raise_for_status()
                hits = parse_efetch(r.text, global_config.PUBMED_ABSTRACT_MAX_CHARS)
        except httpx.HTTPStatusError as e:
            # Never str(e): it embeds the request URL, which carries api_key.
            code = e.response.status_code
            hint = " (rate limited; retry shortly)" if code == 429 else ""
            return _err(f"PubMed returned HTTP {code}{hint}")
        except httpx.HTTPError as e:
            return _err(f"PubMed request failed: {type(e).__name__}")
        except (ValueError, ET.ParseError) as e:
            return _err(f"PubMed returned an unparseable response: {type(e).__name__}")

        # efetch does not promise esearch's order; esearch's is the ranking.
        payload["articles"] = [hits[p] for p in pmids if p in hits]
        if total > len(pmids):
            payload["note"] = (
                f"Showing {len(pmids)} of {total} matches. Abstracts only; "
                "full_text_url (when present) is the free PMC full text."
            )
        return payload
