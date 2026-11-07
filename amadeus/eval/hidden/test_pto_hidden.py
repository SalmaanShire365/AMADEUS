"""Run by the scorer, never shown to the model. Argument: the workspace's tools dir."""

import sys
import unittest
from datetime import date

sys.path.insert(0, sys.argv.pop(1))
from pto_accrual import carryover, monthly_rate  # noqa: E402


class Hidden(unittest.TestCase):
    def test_rate_boundaries(self):
        self.assertEqual(monthly_rate(date(2021, 6, 2), date(2026, 6, 1)), 1.25)
        self.assertEqual(monthly_rate(date(2021, 6, 1), date(2026, 6, 1)), 1.5)
        self.assertEqual(monthly_rate(date(2010, 1, 1), date(2026, 6, 1)), 1.5)

    def test_carryover_matches_handbook(self):
        self.assertEqual(carryover(17), 5)
        self.assertEqual(carryover(5), 5)
        self.assertEqual(carryover(2.5), 2.5)


if __name__ == "__main__":
    unittest.main()
