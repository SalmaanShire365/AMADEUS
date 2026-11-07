import unittest
from datetime import date

from pto_accrual import monthly_rate, years_of_service


class Accrual(unittest.TestCase):
    def test_years_of_service(self):
        self.assertEqual(years_of_service(date(2021, 6, 1), date(2026, 5, 31)), 4)
        self.assertEqual(years_of_service(date(2021, 6, 1), date(2026, 6, 1)), 5)

    def test_base_rate(self):
        self.assertEqual(monthly_rate(date(2024, 1, 1), date(2026, 6, 1)), 1.25)

    def test_senior_rate_starts_on_fifth_anniversary(self):
        self.assertEqual(monthly_rate(date(2021, 6, 1), date(2026, 6, 1)), 1.5)


if __name__ == "__main__":
    unittest.main()
