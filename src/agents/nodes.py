"""Graph nodes for dispute resolution."""

from .state import DisputeState


def load_case(state: DisputeState) -> DisputeState:
    """Load case data into graph state."""
    return state


def resolve_case(state: DisputeState) -> DisputeState:
    """Resolve a case using transaction data and policy evidence."""
    return state
