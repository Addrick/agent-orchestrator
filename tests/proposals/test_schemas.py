# tests/proposals/test_schemas.py
"""Unit tests for the proposal action whitelist (DP-282)."""

from src.proposals.schemas import (
    DISPOSITION_DECISIONS,
    PROPOSAL_ACTIONS,
    build_submission_tool_schema,
    validate_proposal_args,
)


def test_valid_args_for_every_action():
    assert validate_proposal_args("add_note", {"ticket_number": 1, "body": "hi"}) == []
    assert validate_proposal_args("set_priority", {"ticket_number": 1, "priority": "3 high"}) == []
    assert validate_proposal_args("remind", {"ticket_number": 1, "pending_until": "2026-08-01"}) == []


def test_unknown_action_rejected():
    assert validate_proposal_args("delete_ticket", {"ticket_number": 1}) \
        == ["unknown action_type 'delete_ticket'"]


def test_args_must_be_object():
    assert validate_proposal_args("add_note", "ticket 1 note hi") == ["args must be an object"]


def test_missing_required_and_unexpected_args():
    errors = validate_proposal_args("add_note", {"ticket_number": 1, "internal": False})
    assert "missing required argument 'body'" in errors
    # extra keys are rejected, so a proposal can't smuggle args the executor
    # doesn't expect (e.g. flipping the internal flag)
    assert "unexpected argument 'internal'" in errors


def test_type_and_enum_checks():
    assert validate_proposal_args("set_priority", {"ticket_number": "10001", "priority": "3 high"}) \
        == ["argument 'ticket_number' must be int"]
    assert validate_proposal_args("set_priority", {"ticket_number": 1, "priority": "urgent"}) \
        == ["argument 'priority' must be one of ['1 low', '2 normal', '3 high']"]
    # bool is not an acceptable int
    assert validate_proposal_args("set_priority", {"ticket_number": True, "priority": "1 low"}) \
        == ["argument 'ticket_number' must be int"]


def test_date_and_length_checks():
    assert validate_proposal_args("remind", {"ticket_number": 1, "pending_until": "tomorrow"}) \
        == ["argument 'pending_until' must be a YYYY-MM-DD date"]
    errors = validate_proposal_args("add_note", {"ticket_number": 1, "body": "x" * 4001})
    assert errors == ["argument 'body' exceeds 4000 chars"]


def test_submission_tool_schema_tracks_whitelist():
    schema = build_submission_tool_schema()
    assert schema["function"]["name"] == "submit_proposals"
    enum = schema["function"]["parameters"]["properties"]["proposals"]["items"][
        "properties"]["action_type"]["enum"]
    assert set(enum) == set(PROPOSAL_ACTIONS.keys())
    # every whitelisted action is described to the model
    for name in PROPOSAL_ACTIONS:
        assert name in schema["function"]["description"]


# --- DP-290: reflective disposition schema ---

def test_schema_has_no_dispositions_without_pending_ids():
    for schema in (build_submission_tool_schema(),
                   build_submission_tool_schema(pending_ids=None)):
        props = schema["function"]["parameters"]["properties"]
        assert "dispositions" not in props
        assert schema["function"]["parameters"]["required"] == ["proposals"]


def test_schema_dispositions_enum_pinned_to_pending_ids():
    schema = build_submission_tool_schema(pending_ids=[25, 26, 34])
    props = schema["function"]["parameters"]["properties"]
    disp = props["dispositions"]["items"]["properties"]
    # ids the model may address are exactly the injected pending rows
    assert disp["proposal_id"]["enum"] == [25, 26, 34]
    # decisions are a closed enum — no free-form queue operations
    assert disp["decision"]["enum"] == DISPOSITION_DECISIONS
    assert set(props["dispositions"]["items"]["required"]) == {"proposal_id", "decision"}
    # dispositions are optional: an omitted row is left untouched, never destroyed
    assert schema["function"]["parameters"]["required"] == ["proposals"]
    # the new-proposals surface is unchanged
    assert "proposals" in props
