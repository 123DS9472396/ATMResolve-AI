"""Human-readable citation helpers for retrieved evidence."""
from __future__ import annotations

def format_citation(document: dict) -> str:
    """Format a retrieved document using its title and grounded source URL."""
    citation_id = str(document.get("citation_id", "POL"))
    title = str(document.get("title", "Unknown document"))
    source_url = str(document.get("source_url", "unknown source"))
    return f"[{citation_id}] {title} — {source_url}"
