"""CLI tests for the embed stage — verify the commands work."""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch

import pytest

from vtlaw.cli import cmd_embed_status


@pytest.mark.integration
def test_embed_status_reports_coverage():
    """The command should print coverage stats for each label."""
    with patch("sys.stdout", new_callable=StringIO) as mock_out:
        exit_code = cmd_embed_status(None)

    output = mock_out.getvalue()
    assert exit_code == 0
    assert "Article" in output
    assert "Clause" in output
    assert "Point" in output
    assert "embedded" in output
