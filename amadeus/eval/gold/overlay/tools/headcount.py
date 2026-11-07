import csv
from collections import defaultdict

totals = defaultdict(lambda: [0, 0])
with open("employees/roster.csv", newline="") as f:
    for row in csv.DictReader(f):
        if row["status"] != "terminated":
            totals[row["department"]][0] += 1
            totals[row["department"]][1] += int(row["salary_usd"])

with open("reports/headcount_by_department.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["department", "active_count", "total_salary"])
    for dept in sorted(totals):
        w.writerow([dept, *totals[dept]])
