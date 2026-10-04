"""Graph lookup must not connect to PostgreSQL in Lite mode."""

import unittest
from unittest.mock import patch

import graph_store


class GraphStoreLiteTests(unittest.TestCase):
    def test_empty_database_url_falls_back_without_connections(self):
        with patch.object(graph_store, "connection") as connection:
            self.assertIsNone(graph_store.get_ready_graph("", "paper_chunks"))
            self.assertIsNone(
                graph_store.get_ready_graph_for_alias("", object(), "paper_chunks")
            )
            connection.assert_not_called()

    def test_schema_creation_needs_postgres_url(self):
        with self.assertRaises(ValueError):
            graph_store.init_graph_schema("")


if __name__ == "__main__":
    unittest.main()
