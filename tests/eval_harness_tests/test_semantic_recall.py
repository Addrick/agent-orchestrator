"""Unit tests for the semantic recall grading path: resolver + grader."""
from __future__ import annotations

import pytest

from eval_harnesses.framework.grading import SemanticRecallGrader
from eval_harnesses.framework.results import RunOutput
from eval_harnesses.suites.memory_recall.resolver import resolve_facts


class _Scenario:
    def __init__(self, expectations):
        self.expectations = expectations


class FakeClient:
    """Recall returns canned hits keyed by query substring."""

    def __init__(self, corpus):
        # corpus: list of {"id", "text"}
        self.corpus = corpus
        self.calls = []

    async def arecall(self, bank, query, *, max_tokens=None, **_):
        self.calls.append((bank, query))
        q = query.lower()
        # naive: any memory whose text shares >=1 word with query
        q_words = set(w for w in q.split() if len(w) > 3)
        hits = [
            m for m in self.corpus
            if q_words & set(m["text"].lower().split())
        ]
        return hits


CORPUS = [
    {"id": "m_bday", "text": "Adam's birthday is March 14."},
    {"id": "m_year", "text": "Adam was born in 1990."},
    {"id": "m_city", "text": "Adam lives in Portland, Oregon."},
    {"id": "m_color", "text": "Adam's favorite color is dark green."},
]

SEED = {
    "adam_birthday_date": {"text": "Adam's birthday is March 14."},
    "adam_dob_year": {"text": "Adam was born in 1990."},
    "adam_city": {"text": "Adam lives in Portland, Oregon."},
    "adam_fav_color": {"text": "Adam's favorite color is dark green."},
}


@pytest.mark.asyncio
async def test_resolver_resolves_seed_facts():
    facts = [
        {"key": "bday", "source": "seed", "seed_key": "adam_birthday_date"},
        {"key": "year", "source": "seed", "seed_key": "adam_dob_year"},
    ]
    result = await resolve_facts(
        "test_persona", facts, rest_client=FakeClient(CORPUS), seed_data=SEED
    )
    assert result.resolved == {"bday": {"m_bday"}, "year": {"m_year"}}
    assert result.unresolved == []


@pytest.mark.asyncio
async def test_resolver_unresolved_when_seed_missing():
    facts = [{"key": "ghost", "source": "seed", "seed_key": "not_in_seed"}]
    result = await resolve_facts(
        "test_persona", facts, rest_client=FakeClient(CORPUS), seed_data=SEED
    )
    assert result.unresolved == ["ghost"]
    assert "not in seed_data" in result.diagnostics["ghost"]


@pytest.mark.asyncio
async def test_resolver_history_locator():
    facts = [{"key": "bday", "source": "history", "locators": ["March 14"]}]
    result = await resolve_facts(
        "test_persona", facts, rest_client=FakeClient(CORPUS), seed_data=SEED
    )
    assert result.resolved == {"bday": {"m_bday"}}


@pytest.mark.asyncio
async def test_resolver_no_client_marks_all_unresolved():
    facts = [{"key": "bday", "source": "seed", "seed_key": "adam_birthday_date"}]
    result = await resolve_facts(
        "test_persona", facts, rest_client=None, seed_data=SEED
    )
    assert result.unresolved == ["bday"]


def _run_output(retrieved_ids, resolved_map):
    out = RunOutput()
    out.hindsight_hits = [{"id": i, "text": ""} for i in retrieved_ids]
    out.raw = {"resolved_ids": resolved_map}
    return out


def test_grader_passes_when_expected_present_and_no_noise():
    scen = _Scenario({
        "expected_facts": [
            {"key": "bday", "source": "seed", "seed_key": "adam_birthday_date"},
        ],
        "noise_facts": [
            {"key": "color", "source": "seed", "seed_key": "adam_fav_color"},
        ],
    })
    out = _run_output(
        retrieved_ids=["m_bday", "m_year", "m_city"],
        resolved_map={"bday": "m_bday", "color": "m_color"},
    )
    g = SemanticRecallGrader()
    res = g.grade(scen, out)
    assert res.passed
    assert res.detail["per_k"][5]["recall"] == 1.0
    assert res.detail["per_k"][5]["noise_rate"] == 0.0


def test_grader_fails_when_noise_too_high():
    scen = _Scenario({
        "expected_facts": [
            {"key": "bday", "source": "seed", "seed_key": "adam_birthday_date"},
        ],
        "noise_facts": [
            {"key": "color", "source": "seed", "seed_key": "adam_fav_color"},
            {"key": "city", "source": "seed", "seed_key": "adam_city"},
        ],
    })
    out = _run_output(
        retrieved_ids=["m_bday", "m_color", "m_city"],
        resolved_map={"bday": "m_bday", "color": "m_color", "city": "m_city"},
    )
    g = SemanticRecallGrader(noise_threshold=0.25)
    res = g.grade(scen, out)
    # k_pass=5: window has m_bday, m_color, m_city; noise_rate = 2/3 > 0.25
    assert not res.passed
    assert res.detail["per_k"][5]["noise_rate"] > 0.25


def test_grader_multi_fact_recall_denominator():
    scen = _Scenario({
        "expected_facts": [
            {"key": "bday", "source": "seed", "seed_key": "adam_birthday_date"},
            {"key": "year", "source": "seed", "seed_key": "adam_dob_year"},
            {"key": "city", "source": "seed", "seed_key": "adam_city"},
        ],
    })
    out = _run_output(
        retrieved_ids=["m_bday"],
        resolved_map={"bday": "m_bday", "year": "m_year", "city": "m_city"},
    )
    res = SemanticRecallGrader().grade(scen, out)
    # 1 of 3 expected facts hit
    assert res.detail["per_k"][5]["recall"] == pytest.approx(1 / 3)


def test_grader_fails_on_unresolved():
    scen = _Scenario({
        "expected_facts": [
            {"key": "bday", "source": "seed", "seed_key": "adam_birthday_date"},
            {"key": "ghost", "source": "seed", "seed_key": "missing"},
        ],
    })
    out = _run_output(
        retrieved_ids=["m_bday"],
        resolved_map={"bday": "m_bday"},  # ghost missing
    )
    res = SemanticRecallGrader().grade(scen, out)
    assert not res.passed
    assert "UNRESOLVED" in res.notes


def test_grader_mrr():
    scen = _Scenario({
        "expected_facts": [
            {"key": "bday", "source": "seed", "seed_key": "adam_birthday_date"},
        ],
    })
    # expected at index 2 -> mrr = 1/3
    out = _run_output(
        retrieved_ids=["m_city", "m_color", "m_bday"],
        resolved_map={"bday": "m_bday"},
    )
    res = SemanticRecallGrader().grade(scen, out)
    assert res.detail["per_k"][5]["mrr"] == pytest.approx(1 / 3)


def test_grader_id_group_dupes_count_once_for_recall_but_slot_for_precision():
    """A fact with 2 ids: retrieving both counts as 1 fact-hit (recall) but
    eats 2 of K slots (precision)."""
    scen = _Scenario({
        "expected_facts": [
            {"key": "bday", "source": "history", "locators": ["x"]},
        ],
    })
    out = _run_output(
        retrieved_ids=["m_bday_a", "m_bday_b", "m_other"],
        resolved_map={"bday": ["m_bday_a", "m_bday_b"]},
    )
    res = SemanticRecallGrader().grade(scen, out)
    # 1 fact, hit -> recall=1.0
    assert res.detail["per_k"][5]["recall"] == 1.0
    # 2 slots of 3 retrieved are in expected ids -> precision@5 over window=3
    # window has 3 items at K=5 (only 3 retrieved), so precision = 2/3
    assert res.detail["per_k"][5]["precision"] == pytest.approx(2 / 3)


def test_resolver_over_match_unresolved():
    """If a locator matches more than max_matches, fact is unresolved with
    a 'too broad' diagnostic."""
    import asyncio

    corpus = [
        {"id": f"m_{i}", "text": "adam likes things"} for i in range(7)
    ]
    facts = [{
        "key": "broad",
        "source": "history",
        "locators": ["adam"],
        "max_matches": 3,
    }]
    result = asyncio.run(resolve_facts(
        "bank", facts, rest_client=FakeClient(corpus), seed_data={}
    ))
    assert result.unresolved == ["broad"]
    assert "exceeds max_matches" in result.diagnostics["broad"]


def test_resolver_multiple_locators_union():
    """Multiple locators OR together into one id-set."""
    import asyncio

    corpus = [
        {"id": "m1", "text": "Permission denied (publickey)"},
        {"id": "m2", "text": "publickey authentication failed"},
        {"id": "m3", "text": "unrelated"},
    ]
    facts = [{
        "key": "ssh",
        "source": "history",
        "locators": ["Permission denied (publickey)", "publickey authentication failed"],
    }]
    result = asyncio.run(resolve_facts(
        "bank", facts, rest_client=FakeClient(corpus), seed_data={}
    ))
    assert result.resolved == {"ssh": {"m1", "m2"}}


def test_grader_k_sweep_keys():
    scen = _Scenario({
        "expected_facts": [
            {"key": "bday", "source": "seed", "seed_key": "adam_birthday_date"},
        ],
    })
    out = _run_output(
        retrieved_ids=["m_bday"],
        resolved_map={"bday": "m_bday"},
    )
    res = SemanticRecallGrader(k_sweep=(1, 3, 5, 10)).grade(scen, out)
    assert set(res.detail["per_k"].keys()) == {1, 3, 5, 10}
