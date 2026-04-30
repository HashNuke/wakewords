from __future__ import annotations

import unittest

from wakewords.cli import _parse_csv


class CliParsingTests(unittest.TestCase):
    def test_parse_csv_accepts_fire_tuple_values(self) -> None:
        self.assertEqual(_parse_csv(("en", "hi")), ["en", "hi"])

    def test_parse_csv_accepts_comma_separated_string(self) -> None:
        self.assertEqual(_parse_csv("en, hi"), ["en", "hi"])


if __name__ == "__main__":
    unittest.main()
