"""pubmed_search (DP-501): esearch -> efetch, caps, ordering, key hygiene."""

from typing import Any, Dict, List

import httpx
import pytest

from config import global_config
from src.tools.pubmed import PubMedSearchHandler
from src.tools.tool_manager import ToolManager

_EFETCH = """<?xml version="1.0" ?>
<PubmedArticleSet>
 <PubmedArticle>
  <MedlineCitation><PMID>111</PMID><Article>
   <Journal><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue>
    <Title>Nature</Title></Journal>
   <ArticleTitle>Gene <i>editing</i> works</ArticleTitle>
   <Abstract>
    <AbstractText Label="BACKGROUND">Some background.</AbstractText>
    <AbstractText Label="RESULTS">It worked.</AbstractText>
   </Abstract>
   <AuthorList>
    <Author><LastName>Smith</LastName><Initials>J</Initials></Author>
    <Author><CollectiveName>The CRISPR Consortium</CollectiveName></Author>
   </AuthorList>
  </Article></MedlineCitation>
  <PubmedData><ArticleIdList>
   <ArticleId IdType="pubmed">111</ArticleId>
   <ArticleId IdType="doi">10.1/abc</ArticleId>
   <ArticleId IdType="pmc">PMC999</ArticleId>
  </ArticleIdList></PubmedData>
 </PubmedArticle>
 <PubmedArticle>
  <MedlineCitation><PMID>222</PMID><Article>
   <Journal><JournalIssue><PubDate><MedlineDate>1998 Jan-Feb</MedlineDate></PubDate>
    </JournalIssue><Title>Old J</Title></Journal>
   <ArticleTitle>No abstract here</ArticleTitle>
  </Article></MedlineCitation>
 </PubmedArticle>
</PubmedArticleSet>"""


def _handler(requests: List[httpx.Request], esearch: Dict[str, Any],
             efetch: str = _EFETCH, status: int = 200) -> PubMedSearchHandler:
    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if status != 200:
            return httpx.Response(status)
        if request.url.path.endswith("esearch.fcgi"):
            return httpx.Response(200, json={"esearchresult": esearch})
        return httpx.Response(200, text=efetch)
    return PubMedSearchHandler(transport=httpx.MockTransport(respond))


async def _run(handler: PubMedSearchHandler, **kwargs: Any) -> Dict[str, Any]:
    manager = ToolManager()
    handler.register(manager)
    out = await manager.execute_tool("pubmed_search", {"pubmed_search"}, **kwargs)
    return out["result"]


@pytest.mark.asyncio
async def test_parses_and_keeps_esearch_ranking():
    reqs: List[httpx.Request] = []
    # esearch ranks 222 first; efetch returns 111 first.
    h = _handler(reqs, {"count": "40", "idlist": ["222", "111"], "querytranslation": "q"})
    res = await _run(h, query="crispr")
    assert res["status"] == "ok"
    assert [a["pmid"] for a in res["articles"]] == ["222", "111"]
    a = res["articles"][1]
    assert a["title"] == "Gene editing works"
    assert a["authors"] == ["Smith J", "The CRISPR Consortium"]
    assert a["year"] == "2024" and a["journal"] == "Nature"
    assert a["doi"] == "10.1/abc" and a["pmcid"] == "PMC999"
    assert a["full_text_url"].endswith("/PMC999/")
    assert a["abstract"] == "BACKGROUND: Some background.\nRESULTS: It worked."
    old = res["articles"][0]
    assert old["year"] == "1998" and old["abstract"] is None and old["doi"] is None
    assert res["total_matches"] == 40 and "Showing 2 of 40" in res["note"]
    assert reqs[1].url.params["id"] == "222,111"


@pytest.mark.asyncio
@pytest.mark.parametrize("asked", [500, "ten"])
async def test_max_results_goes_through_the_ceiling(monkeypatch, asked):
    monkeypatch.setattr(global_config, "PUBMED_SEARCH_LIMIT_MAX", 3)
    reqs: List[httpx.Request] = []
    await _run(_handler(reqs, {"count": "0", "idlist": []}), query="x", max_results=asked)
    assert int(reqs[0].url.params["retmax"]) <= 3


@pytest.mark.asyncio
async def test_long_abstract_is_cut_and_flagged(monkeypatch):
    monkeypatch.setattr(global_config, "PUBMED_ABSTRACT_MAX_CHARS", 10)
    res = await _run(_handler([], {"count": "1", "idlist": ["111"]}), query="x")
    a = res["articles"][0]
    assert len(a["abstract"]) == 10 and a["abstract_truncated"] is True


@pytest.mark.asyncio
async def test_zero_hits_skips_efetch_and_explains():
    reqs: List[httpx.Request] = []
    res = await _run(_handler(reqs, {"count": "0", "idlist": []}), query="zzz")
    assert res["articles"] == [] and "No matches" in res["note"]
    assert len(reqs) == 1


@pytest.mark.asyncio
async def test_date_sort_maps_to_pub_date():
    reqs: List[httpx.Request] = []
    await _run(_handler(reqs, {"count": "0", "idlist": []}), query="x", sort="date")
    assert reqs[0].url.params["sort"] == "pub_date"


@pytest.mark.asyncio
async def test_api_key_sent_but_never_echoed_in_errors(monkeypatch):
    monkeypatch.setenv("NCBI_API_KEY", "sekritkey123")
    reqs: List[httpx.Request] = []
    res = await _run(_handler(reqs, {}, status=429), query="x")
    assert reqs[0].url.params["api_key"] == "sekritkey123"
    assert res["status"] == "error" and "429" in res["error"]
    assert "sekritkey123" not in str(res)


@pytest.mark.asyncio
async def test_no_api_key_param_when_unset(monkeypatch):
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    reqs: List[httpx.Request] = []
    await _run(_handler(reqs, {"count": "0", "idlist": []}), query="x")
    assert "api_key" not in reqs[0].url.params
    assert reqs[0].url.params["tool"] == "derpr"
