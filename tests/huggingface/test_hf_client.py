"""Unit tests for the HuggingFace Hub client (DP-265).

No network: the HTTP layer is the thin part, so these pin the parts that decide
what gets installed — which strings are accepted as a repo/file, which tree rows
become an installable file, and which byte count is believed.
"""

from __future__ import annotations

import pytest

from src.huggingface.client import (
    HFClient,
    HFError,
    HFFile,
    _MAX_SEARCH_TAGS,
    _next_cursor,
    _select_tags,
    validate_file_path,
    validate_repo_id,
)


# -- input validation --------------------------------------------------------

@pytest.mark.parametrize("repo", [
    "TheBloke/Model-GGUF",
    "unsloth/gemma-4-31b-it-GGUF",
    "a/b",
    "owner.name/model_v2-Q6",
])
def test_valid_repo_ids_pass(repo):
    assert validate_repo_id(repo) == repo


@pytest.mark.parametrize("repo", [
    "",
    "no-slash",
    "owner/name/extra",
    "../../etc/passwd",
    "owner/../secret",
    "/leading/slash",
    "owner/name?x=1",
    "owner name/model",
])
def test_traversal_and_malformed_repo_ids_are_refused(repo):
    """The repo id is the one place a model steers a URL, so it is validated
    rather than merely percent-encoded — an encoded traversal comes back 404,
    and a 404 reads to the model as 'try another spelling'."""
    with pytest.raises(HFError):
        validate_repo_id(repo)


@pytest.mark.parametrize("path", [
    "model-Q6_K.gguf",
    "quants/model-Q4_K_M.gguf",
])
def test_valid_file_paths_pass(path):
    assert validate_file_path(path) == path


@pytest.mark.parametrize("path", ["../x.gguf", "a/../../b.gguf", "/abs.gguf", ""])
def test_traversal_file_paths_are_refused(path):
    with pytest.raises(HFError):
        validate_file_path(path)


# -- tree row parsing --------------------------------------------------------

def test_lfs_size_wins_over_the_pointer_size():
    """An LFS row carries the pointer's ~135 bytes at top level on some repos.
    Believing that number would size the node's free-space precheck three orders
    of magnitude too small — the one direction that fills a thin pool."""
    row = {
        "type": "file",
        "path": "model-Q6_K.gguf",
        "size": 135,
        "lfs": {"oid": "a" * 64, "size": 24_000_000_000, "pointerSize": 135},
    }
    entry = HFClient._as_gguf_file(row)
    assert entry is not None
    assert entry.size_bytes == 24_000_000_000
    assert entry.sha256 == "a" * 64


def test_non_lfs_file_has_no_sha256():
    row = {"type": "file", "path": "small.gguf", "size": 4096}
    entry = HFClient._as_gguf_file(row)
    assert entry is not None
    assert entry.sha256 is None


def test_non_gguf_and_directories_are_skipped():
    assert HFClient._as_gguf_file({"type": "file", "path": "README.md", "size": 1}) is None
    assert HFClient._as_gguf_file({"type": "directory", "path": "quants"}) is None
    assert HFClient._as_gguf_file("not a dict") is None


def test_a_non_sha256_oid_is_dropped_rather_than_passed_through():
    """A git blob sha1 in `lfs.oid` must not be mistaken for a digest — the node
    would then verify a 40-hex value that can never match sha256sum output."""
    row = {"type": "file", "path": "m.gguf", "size": 10, "lfs": {"oid": "b" * 40, "size": 10}}
    entry = HFClient._as_gguf_file(row)
    assert entry is not None and entry.sha256 is None


def test_unparseable_size_drops_the_row():
    row = {"type": "file", "path": "m.gguf", "size": "big"}
    assert HFClient._as_gguf_file(row) is None


# -- pagination --------------------------------------------------------------

def test_next_cursor_read_from_link_header():
    header = '<https://huggingface.co/api/models/a/b/tree/main?cursor=ZXlKbQ%3D%3D>; rel="next"'
    assert _next_cursor(header) == "ZXlKbQ%3D%3D"


def test_no_next_link_ends_the_walk():
    assert _next_cursor(None) is None
    assert _next_cursor('<https://x/prev>; rel="prev"') is None


# -- find_gguf_file ----------------------------------------------------------

@pytest.mark.asyncio
async def test_find_refuses_a_file_with_no_published_digest(monkeypatch):
    """The digest is the only thing tying the bytes that land on the node to the
    bytes a human approved, so 'install it unverified' is not a degraded mode —
    it is the whole control being switched off."""
    client = HFClient()

    async def fake_list(repo, revision="main"):
        return [HFFile(path="m.gguf", size_bytes=10, sha256=None)]

    monkeypatch.setattr(client, "list_gguf_files", fake_list)
    with pytest.raises(HFError, match="no LFS sha256"):
        await client.find_gguf_file("a/b", "m.gguf")


@pytest.mark.asyncio
async def test_find_lists_what_the_repo_does_offer_on_a_miss(monkeypatch):
    client = HFClient()

    async def fake_list(repo, revision="main"):
        return [HFFile(path="real-Q6.gguf", size_bytes=10, sha256="c" * 64)]

    monkeypatch.setattr(client, "list_gguf_files", fake_list)
    with pytest.raises(HFError, match="real-Q6.gguf"):
        await client.find_gguf_file("a/b", "typo-Q6.gguf")


@pytest.mark.asyncio
async def test_find_returns_the_matching_entry(monkeypatch):
    client = HFClient()
    wanted = HFFile(path="m-Q6.gguf", size_bytes=99, sha256="d" * 64)

    async def fake_list(repo, revision="main"):
        return [HFFile(path="other.gguf", size_bytes=1, sha256="e" * 64), wanted]

    monkeypatch.setattr(client, "list_gguf_files", fake_list)
    assert await client.find_gguf_file("a/b", "m-Q6.gguf") is wanted


def test_to_dict_reports_bytes_and_gib():
    entry = HFFile(path="m.gguf", size_bytes=2 * 1024 ** 3, sha256="f" * 64)
    assert entry.to_dict() == {
        "path": "m.gguf",
        "size_bytes": 2147483648,
        "size_gib": 2.0,
        "sha256": "f" * 64,
    }


# -- search payload ----------------------------------------------------------

def test_search_tags_keep_base_model_ahead_of_the_truncation():
    """`base_model:` survives the tag cap (DP-335 review).

    The tool description, the handler note and hypr's prompt all tell the model
    to match a quant repo to its upstream model by this tag. The cap used to be
    a bare slice over the Hub's own unordered list, and `base_model:` sorts
    late — so a model told three times to read the field searched hits that did
    not carry it, concluded no quant corresponded to the model it was asked
    about, and re-queried: the exact budget-burning loop DP-335 exists to
    break.
    """
    raw = (
        ["gguf", "transformers", "text-generation", "conversational"]
        + [f"lang:{c}" for c in "abcdefghij"]
        + ["base_model:Qwen/Qwen3.8-27B", "license:apache-2.0"]
    )

    tags = _select_tags(raw)

    assert len(tags) == _MAX_SEARCH_TAGS
    assert tags[0] == "base_model:Qwen/Qwen3.8-27B"
    # Everything else keeps the Hub's own order, so a truncated list still
    # reads like the source.
    assert tags[1:4] == ["gguf", "transformers", "text-generation"]


def test_search_tags_are_unchanged_when_there_is_no_base_model_tag():
    tags = _select_tags(["gguf", "transformers"])
    assert tags == ["gguf", "transformers"]


def test_search_tags_tolerate_a_missing_or_non_string_tag_list():
    assert _select_tags(None) == []
    assert _select_tags([1, "gguf"]) == ["1", "gguf"]
