"""Compute reviewer agreement on the risk taxonomy once review sheets come back.

Usage:
    python review/agreement.py reviewer1.csv [reviewer2.csv ...]

Each CSV must be a filled copy of risk_taxonomy_review_sheet.csv. The script
reads the reviewer's level from the "If N, reviewer's level" column when the
"Reviewer agrees? (Y/N)" column says N, and otherwise takes our level as the
reviewer's. It reports:

  * percentage agreement with our taxonomy, per reviewer
  * Cohen's kappa (one reviewer vs us; two reviewers vs each other)
  * Fleiss' kappa (three or more raters, including us)
  * the specific categories that were disputed

Chance-corrected agreement matters here because the levels are not uniformly
distributed -- 22 of 41 categories are Medium, so a rater who answered "Medium"
every time would score 54% raw agreement while carrying no information.
"""
import sys, csv, itertools
from collections import Counter

LEVELS = ["High", "Medium", "Low"]

def load(path):
    ours, theirs, cats = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            cat = (row.get("Clause category") or "").strip()
            if not cat: continue
            our = (row.get("Our risk level (High/Medium/Low)") or "").strip()
            agree = (row.get("Reviewer agrees? (Y/N)") or "").strip().upper()
            alt = (row.get("If N, reviewer's level") or "").strip()
            if agree.startswith("N") and alt:
                rev = alt.capitalize()
            elif agree.startswith("Y"):
                rev = our
            else:
                continue                      # unanswered row: excluded
            cats.append(cat); ours.append(our); theirs.append(rev)
    return cats, ours, theirs

def cohen_kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l]/n) * (cb[l]/n) for l in LEVELS)
    return (po - pe) / (1 - pe) if pe != 1 else float("nan"), po

def fleiss_kappa(rating_lists):
    n_items = len(rating_lists[0]); n_raters = len(rating_lists)
    counts = [[sum(r[i] == l for r in rating_lists) for l in LEVELS] for i in range(n_items)]
    p_i = [(sum(c*c for c in row) - n_raters) / (n_raters*(n_raters-1)) for row in counts]
    p_bar = sum(p_i)/n_items
    p_j = [sum(row[j] for row in counts)/(n_items*n_raters) for j in range(len(LEVELS))]
    pe = sum(p*p for p in p_j)
    return (p_bar - pe)/(1 - pe) if pe != 1 else float("nan")

def interpret(k):
    for lo, label in [(0.81,"almost perfect"),(0.61,"substantial"),(0.41,"moderate"),
                      (0.21,"fair"),(0.0,"slight")]:
        if k >= lo: return label
    return "poor (worse than chance)"

if len(sys.argv) < 2:
    print(__doc__); sys.exit(1)

sheets = [load(p) for p in sys.argv[1:]]
cats, ours = sheets[0][0], sheets[0][1]
print(f"categories scored: {len(cats)} of 41\n")

all_ratings = [ours]
for path, (c, o, t) in zip(sys.argv[1:], sheets):
    k, po = cohen_kappa(o, t)
    disagreed = [(cc, oo, tt) for cc, oo, tt in zip(c, o, t) if oo != tt]
    print(f"{path}")
    print(f"  agreement with our taxonomy : {po:.1%} ({len(c)-len(disagreed)}/{len(c)})")
    print(f"  Cohen's kappa vs us         : {k:.3f}  ({interpret(k)})")
    if disagreed:
        print(f"  disputed ({len(disagreed)}):")
        for cc, oo, tt in disagreed:
            print(f"      {cc:<38s} ours={oo:<7s} reviewer={tt}")
    print()
    all_ratings.append(t)

if len(sheets) >= 2:
    for (p1, s1), (p2, s2) in itertools.combinations(zip(sys.argv[1:], sheets), 2):
        k, po = cohen_kappa(s1[2], s2[2])
        print(f"reviewer-vs-reviewer {p1} / {p2}: agreement {po:.1%}, Cohen's kappa {k:.3f} ({interpret(k)})")

if len(all_ratings) >= 3:
    k = fleiss_kappa(all_ratings)
    print(f"\nFleiss' kappa over {len(all_ratings)} raters (us + {len(all_ratings)-1} reviewers): "
          f"{k:.3f}  ({interpret(k)})")

print("\nReport the disagreement count and kappa in Section 4.4.2 WHATEVER the result is.")
