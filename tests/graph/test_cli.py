"""Graph rebuild commands must leave the temporal graph usable."""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vtlaw.cli import cmd_graph_import


def test_graph_import_loads_amendments_with_the_snapshot():
    parse_stats = SimpleNamespace(failed=[], provisions=3, summary=lambda: "3 provisions")
    import_stats = SimpleNamespace(
        failed=[], missing_parent=[], summary=lambda: "3 provisions"
    )
    counts = SimpleNamespace(provisions=3, summary=lambda: "3 provisions")
    amend_stats = SimpleNamespace(summary=lambda: "imported=1, skipped=0")
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None

    with (
        patch("vtlaw.cli.parse_corpus", return_value=([MagicMock()], parse_stats)),
        patch("vtlaw.cli._graph_client", return_value=client),
        patch("vtlaw.cli.import_documents", return_value=import_stats),
        patch("vtlaw.cli.count_graph", return_value=counts),
        patch("vtlaw.cli.import_amends_directory", create=True, return_value=amend_stats) as amends,
    ):
        result = cmd_graph_import(Namespace(snapshot="data/snapshot", wipe=False))

    assert result == 0
    amends.assert_called_once_with(client, "data/amends")
