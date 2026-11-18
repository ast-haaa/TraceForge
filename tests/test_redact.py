"""Unit tests for agentreplay.redact."""
import json

from agentreplay.models import Event, EventKind, Step, Trace
from agentreplay.redact import redact_sensitive
from agentreplay.sdk import Recorder


def test_secret_keys_are_removed():
    data = {"name": "Alice", "api_key": "sk-secret123", "email": "a@b.com"}
    result = redact_sensitive(data)
    assert "api_key" not in result
    assert result["name"] == "Alice"


def test_case_insensitive_key_removal():
    for key in ("API_KEY", "Api_Key", "PASSWORD", "Token", "access_token"):
        result = redact_sensitive({key: "super-secret"})
        assert key not in result, f"expected key {key!r} to be removed"


def test_email_redaction_in_string():
    result = redact_sensitive("Contact alice@example.com for help")
    assert "alice@example.com" not in result
    assert "<REDACTED_EMAIL>" in result


def test_phone_redaction_in_string():
    result = redact_sensitive("Call us at 555-867-5309")
    assert "5309" not in result
    assert "<REDACTED_PHONE>" in result


def test_nested_dict_redaction():
    data = {"user": {"name": "Bob", "token": "tok_xyz", "phone": "555-123-4567"}}
    result = redact_sensitive(data)
    assert "token" not in result["user"]
    assert "<REDACTED_PHONE>" in result["user"]["phone"]


def test_list_elements_are_redacted():
    data = [{"api_key": "bad"}, "john@doe.com"]
    result = redact_sensitive(data)
    assert "api_key" not in result[0]
    assert "<REDACTED_EMAIL>" in result[1]


def test_none_passthrough():
    assert redact_sensitive(None) is None


def test_non_secret_keys_preserved():
    data = {"order_id": "O100", "amount": 49.99, "node_name": "refund"}
    result = redact_sensitive(data)
    assert result == data


def test_no_mutation_of_original():
    original = {"api_key": "s3cr3t", "name": "Alice"}
    _ = redact_sensitive(original)
    assert "api_key" in original  # original dict is NOT mutated


def test_redaction_across_every_stored_database_column(db_session):
    """Verify API key, email, and phone number never appear in any DB column:
    raw_input, raw_output, output, state_before, state_after, Trace.final_output.
    """
    secret_key = "sk-live-super-secret-key-99999"
    secret_email = "victim.private@forbidden-domain.com"
    secret_phone = "555-987-6543"

    payload = {
        "api_key": secret_key,
        "email": secret_email,
        "phone": secret_phone,
        "text": f"User key={secret_key}, mail={secret_email}, tel={secret_phone}",
    }

    rec = Recorder(db=db_session, run_id="redact-full-audit", agent_name="audit_agent")
    step = rec.start_step("audit_node", state_before=payload)
    event = rec.record_event(
        kind=EventKind.TOOL_CALL,
        raw_input=payload,
        output=payload,
    )
    rec.end_step(state_after=payload)
    rec.finalize(final_output=payload)

    # Fetch stored records fresh from database
    stored_trace = db_session.query(Trace).filter_by(id=rec.trace.id).one()
    stored_step = db_session.query(Step).filter_by(id=step.id).one()
    stored_event = db_session.query(Event).filter_by(id=event.id).one()

    columns_to_check = {
        "raw_input": stored_event.raw_input,
        "raw_output": stored_event.raw_output,
        "output": json.dumps(stored_event.output),
        "state_before": json.dumps(stored_step.state_before),
        "state_after": json.dumps(stored_step.state_after),
        "Trace.final_output": json.dumps(stored_trace.final_output),
    }

    secrets = [secret_key, secret_email, secret_phone]

    for col_name, col_value in columns_to_check.items():
        assert col_value is not None, f"{col_name} was unexpectedly None"
        for secret in secrets:
            assert secret not in col_value, (
                f"Secret leak in column '{col_name}': found {secret!r} in {col_value}"
            )
