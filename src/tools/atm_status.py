"""Read-only synthetic ATM event lookup tool. Never connects to an ATM network."""
from src.storage.repository import get_atm_events


def get_atm_status_events(transaction_id: str) -> list[dict]:
    """Return fictional ATM events for a demo transaction reference."""
    if not transaction_id or len(transaction_id) > 40:
        return []
    return get_atm_events(transaction_id.strip().upper())
