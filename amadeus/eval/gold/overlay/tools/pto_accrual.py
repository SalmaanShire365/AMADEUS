"""PTO helpers used for the year-end balance report."""

from datetime import date

BASE_DAYS_PER_MONTH = 1.25
SENIOR_DAYS_PER_MONTH = 1.5
SENIOR_YEARS = 5
CARRYOVER_LIMIT = 5


def years_of_service(start: date, as_of: date) -> int:
    years = as_of.year - start.year
    if (as_of.month, as_of.day) < (start.month, start.day):
        years -= 1
    return years


def monthly_rate(start: date, as_of: date) -> float:
    """Employees reach the senior rate on their fifth anniversary."""
    if years_of_service(start, as_of) >= SENIOR_YEARS:
        return SENIOR_DAYS_PER_MONTH
    return BASE_DAYS_PER_MONTH


def carryover(balance: float) -> float:
    """Days that carry into the new year."""
    return min(balance, CARRYOVER_LIMIT)
