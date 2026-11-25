import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from pokemon_rag.ingest_expanded import Downloader, annotate_sections, catalog, ingest, structured_sections


class CatalogTests(unittest.TestCase):
    def test_first_three_badges_and_optional_full_walkthrough(self):
        initial = catalog()
        later = catalog(include_later=True)
        self.assertEqual({s["part"] for s in initial if "part" in s}, {4, 5, 6})
        self.assertEqual({s["part"] for s in later if "part" in s}, set(range(4, 18)))
        self.assertTrue(any(s["site"] == "strategywiki" for s in initial))

    def test_source_metadata_does_not_claim_other_generations_are_gen_one(self):
        source = {"id": "center", "site": "bulbapedia", "category": "mechanics", "title": "Pokémon Center"}
        sections = [{"section": name, "text": "content", "location": "", "part": None, "source": "url"}
                    for name in ("Introduction", "In the core series games > Generation I",
                                 "In the core series games > Generation II", "In the spin-off games > Pokémon GO")]
        output = annotate_sections(source, sections)
        self.assertEqual(len(output), 2)
        self.assertTrue(all(row["generation"] is None for row in output))
        self.assertTrue(all(row["scope"] == "mixed games and generations" for row in output))

    def test_curated_facts_include_provenance(self):
        records = structured_sections()
        self.assertGreaterEqual(len(records), 4)
        self.assertTrue(all(r["source"].startswith("https://") and r["source_section"] for r in records))


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name)
        self.source = {"id": "article", "site": "bulbapedia", "url": "https://example.org/wiki/article"}

    @patch("pokemon_rag.ingest_expanded.time.sleep")
    @patch("pokemon_rag.ingest_expanded.requests.get")
    def test_http_403_stops_without_caching_or_retry(self, get, sleep):
        robots = Mock(text="User-agent: *\nAllow: /\n")
        robots.raise_for_status.return_value = None
        denied = Mock()
        denied.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")
        get.side_effect = [robots, denied]
        with patch("pokemon_rag.ingest_expanded.DATA_DIR", self.data):
            with self.assertRaises(requests.HTTPError):
                Downloader().fetch(self.source)
        self.assertEqual(get.call_count, 2)
        self.assertFalse((self.data / "raw/bulbapedia/article.html").exists())

    @patch("pokemon_rag.ingest_expanded.requests.get")
    def test_offline_uses_manual_html_without_network(self, get):
        path = self.data / "raw/bulbapedia/article.html"
        path.parent.mkdir(parents=True)
        path.write_text("manually saved page", encoding="utf-8")
        with patch("pokemon_rag.ingest_expanded.DATA_DIR", self.data):
            self.assertEqual(Downloader(offline=True).fetch(self.source), "manually saved page")
        get.assert_not_called()

    def test_does_not_replace_existing_output(self):
        with self.assertRaises(FileExistsError):
            ingest({"bulbapedia"}, self.data)

    @patch("pokemon_rag.ingest_expanded.requests.get")
    def test_offline_strategywiki_html_is_processed_with_attribution(self, get):
        (self.data / "sections.json").write_text("[]", encoding="utf-8")
        path = self.data / "raw/strategywiki/fake.html"
        path.parent.mkdir(parents=True)
        path.write_text('<div class="mw-parser-output"><h2>Route 2</h2><p>Proceed north.</p></div>')
        source = {"id": "fake", "site": "strategywiki", "category": "walkthroughs",
                  "title": "Route 2", "url": "https://strategywiki.org/wiki/example"}
        output = self.data / "output"
        with patch("pokemon_rag.ingest_expanded.DATA_DIR", self.data), \
             patch("pokemon_rag.ingest_expanded.catalog", return_value=[source]):
            ingest({"strategywiki"}, output, offline=True)
        rows = json.loads((output / "processed/walkthroughs.json").read_text())
        self.assertEqual(rows[0]["source"], source["url"])
        self.assertEqual(rows[0]["text"], "Proceed north.")
        self.assertIn("Yellow", rows[0]["game"])
        get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
