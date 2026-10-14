"""Shared pytest fixtures for the ATMResolve AI test suite.

The most important fixture here is ``isolated_db``, which redirects the
application database to a temporary file for the duration of each test that
requests it.  Tests that call ``init_db()`` or any repository function should
include this fixture to prevent writes to the real ``data/atmresolve.db``.
"""
from __future__ import annotations

import pytest
import src.config as _config
import src.storage.repository as _repo


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    """Redirect APPLICATION_DB_PATH to a fresh temporary file for one test.

    The fixture patches both the config module and the repository module so that
    every call to ``get_connection()`` uses the temp file.  The real database is
    never touched.
    """
    temp_db = tmp_path / "test_atmresolve.db"
    monkeypatch.setattr(_config, "APPLICATION_DB_PATH", temp_db)
    monkeypatch.setattr(_repo, "APPLICATION_DB_PATH", temp_db)
    _repo.init_db()
    return temp_db
