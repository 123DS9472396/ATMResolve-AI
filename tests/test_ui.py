"""Tests for Streamlit UI utility functions and health monitor helpers."""
from __future__ import annotations

import pytest
from app.ui_components import format_iso_timestamp, fetch_health


def test_format_iso_timestamp_valid_offsets():
    """Valid ISO timestamps with timezones must be formatted cleanly."""
    res = format_iso_timestamp("2026-10-14T10:15:00+05:30")
    assert res == "2026-10-14 10:15:00"


def test_format_iso_timestamp_z_suffix():
    """Valid UTC ISO timestamps with 'Z' suffix must be parsed without error."""
    res = format_iso_timestamp("2026-10-14T10:15:00Z")
    assert res == "2026-10-14 10:15:00"


def test_format_iso_timestamp_empty_or_none():
    """Empty or None timestamp input must return an em dash fallback."""
    assert format_iso_timestamp(None) == "—"
    assert format_iso_timestamp("") == "—"


def test_format_iso_timestamp_invalid_string():
    """Malformed timestamp strings should gracefully fall back to string representation."""
    res = format_iso_timestamp("invalid-date-format")
    assert res == "invalid-date-format"


def test_fetch_health_returns_expected_keys():
    """fetch_health() must always return a dictionary with required health status keys."""
    health = fetch_health()
    assert isinstance(health, dict)
    assert "reachable" in health
    assert "status" in health
    assert "mode" in health
    assert "components" in health
    assert isinstance(health["reachable"], bool)
    assert health["status"] in {"ok", "degraded", "unreachable"}


def test_ui_components_renderers_are_callable():
    """Key UI component renderers must be importable callables."""
    import app.ui_components as ui
    assert callable(ui.render_sidebar_health)
    assert callable(ui.render_overview)
    assert callable(ui.render_new_investigation)
    assert callable(ui.render_case_review)
    assert callable(ui.render_audit_log_page)
    assert callable(ui.render_policy_search)
    assert callable(ui.render_evaluation)
