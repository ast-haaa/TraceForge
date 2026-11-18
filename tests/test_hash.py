"""Unit tests for agentreplay.hash."""
from datetime import date

from agentreplay.hash import compute_input_hash, strip_volatile_fields


def test_hash_is_deterministic():
    h1 = compute_input_hash({"b": 2, "a": 1})
    h2 = compute_input_hash({"b": 2, "a": 1})
    assert h1 == h2


def test_hash_is_key_order_independent():
    h1 = compute_input_hash({"b": 2, "a": 1})
    h2 = compute_input_hash({"a": 1, "b": 2})
    assert h1 == h2, "dict key order must not affect hash"


def test_hash_is_content_sensitive():
    assert compute_input_hash({"a": 1}) != compute_input_hash({"a": 2})
    assert compute_input_hash({"a": 1}) != compute_input_hash({"b": 1})


def test_hash_on_nested_structures():
    obj = {"prompt": "hello", "args": {"order_id": "O100", "amount": 49.99}}
    h = compute_input_hash(obj)
    assert isinstance(h, str) and len(h) == 64  # SHA-256 hex


def test_hash_on_list_input():
    h1 = compute_input_hash([1, 2, 3])
    h2 = compute_input_hash([1, 2, 3])
    assert h1 == h2


def test_hash_list_order_matters():
    # Lists are ordered – different order means different hash
    assert compute_input_hash([1, 2]) != compute_input_hash([2, 1])


def test_order_date_objects_produce_different_hashes():
    """datetime.date objects represent real domain content and must NOT be stripped."""
    payload1 = {"order_date": date(2026, 11, 6)}
    payload2 = {"order_date": date(2026, 10, 25)}
    assert compute_input_hash(payload1) != compute_input_hash(payload2)


def test_date_strings_produce_different_hashes():
    """String date fields (e.g. 'date': 'YYYY-MM-DD') must NOT be stripped."""
    payload1 = {"date": "2026-11-06"}
    payload2 = {"date": "2026-10-25"}
    assert compute_input_hash(payload1) != compute_input_hash(payload2)


def test_same_payload_with_different_issued_at_request_id_uuid_produces_same_hash():
    """Clearly volatile fields (issued_at, request_id, uuid) must yield identical hashes."""
    run1 = {
        "customer_id": "C001",
        "order_id": "O100",
        "issued_at": "2026-11-18T10:00:00.123456+00:00",
        "request_id": "11111111-2222-3333-4444-555555555555",
        "uuid": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    }
    run2 = {
        "customer_id": "C001",
        "order_id": "O100",
        "issued_at": "2026-11-18T12:34:56.987654+00:00",
        "request_id": "99999999-8888-7777-6666-000000000000",
        "uuid": "12345678-1234-1234-1234-123456789abc",
    }
    assert compute_input_hash(run1) == compute_input_hash(run2)


def test_strings_differing_only_in_full_iso_timestamp_produce_same_hash():
    """String payloads differing only in a full ISO timestamp string normalize to same hash."""
    msg1 = "Processed order O100 at 2026-11-18T10:00:00Z successfully"
    msg2 = "Processed order O100 at 2026-11-18T15:30:45.123456+00:00 successfully"
    assert compute_input_hash(msg1) == compute_input_hash(msg2)


def test_different_real_content_produces_different_hash():
    """Different real domain data must produce different hashes."""
    base = {
        "customer_id": "C001",
        "order_id": "O100",
        "request_id": "11111111-2222-3333-4444-555555555555",
        "timestamp": "2026-11-18T10:00:00Z",
    }
    different_order = {
        "customer_id": "C001",
        "order_id": "O101",  # different real content
        "request_id": "11111111-2222-3333-4444-555555555555",
        "timestamp": "2026-11-18T10:00:00Z",
    }
    assert compute_input_hash(base) != compute_input_hash(different_order)


def test_strip_volatile_fields_preserves_clean_data():
    clean_data = {"name": "Alice", "amount": 49.99}
    assert strip_volatile_fields(clean_data) == clean_data
