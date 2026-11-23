"""Test of experiment/make_subset.py on a tiny dummy raw file (no model needed).

Run from the repo root:
    python -m unittest discover tests
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "experiment" / "make_subset.py"


class TestMakeSubset(unittest.TestCase):
    def test_keeps_only_complete_facts_and_adds_relation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # fact 0 has all 19 variants, fact 1 only 5 (like a run stopped in the middle of a fact)
            rows = [{"id": f"fact-{f:04d}__dim__cat{i:02d}", "fact_index": f,
                     "dimension": "dim", "category": f"cat{i:02d}"}
                    for f, n in [(0, 19), (1, 5)] for i in range(n)]
            pd.DataFrame(rows).to_parquet(tmp / "raw.parquet")
            pd.DataFrame({"id": [r["id"] for r in rows],
                          "relation": ["capital" if r["fact_index"] == 0 else "author" for r in rows]}
                         ).to_csv(tmp / "data.csv", index=False)

            subprocess.run([sys.executable, str(SCRIPT), "--raw", str(tmp / "raw.parquet"),
                            "--data", str(tmp / "data.csv"), "--out", str(tmp / "subset.parquet")],
                           check=True, capture_output=True)

            out = pd.read_parquet(tmp / "subset.parquet")
            self.assertEqual(len(out), 19)
            self.assertEqual(set(out.fact_index), {0})
            self.assertEqual(set(out.relation), {"capital"})
            self.assertTrue((tmp / "subset.csv").exists())


if __name__ == "__main__":
    unittest.main()
