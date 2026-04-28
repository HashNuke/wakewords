from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path


class PlaygroundMfccTests(unittest.TestCase):
    def test_javascript_mfcc_matches_nemo_fixtures(self) -> None:
        if shutil.which("node") is None:
            self.skipTest("node is required for playground MFCC parity test")

        subprocess.run(
            ["node", str(Path(__file__).with_suffix(".mjs"))],
            check=True,
            cwd=Path(__file__).resolve().parents[1],
        )


if __name__ == "__main__":
    unittest.main()
