"""Read-only synthetic transaction lookup tool. Never connects to a real bank."""
from src.storage.repository import get_transaction


def lookup_transaction(transaction_id: str) -> dict | None:
    """Return one fictional transaction by demo reference, or None if absent."""
    if not transaction_id or len(transaction_id) > 40:
        return None
    return get_transaction(transaction_id.strip().upper())
