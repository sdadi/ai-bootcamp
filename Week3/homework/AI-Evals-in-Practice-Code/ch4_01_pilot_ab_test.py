"""
Chapter 4.1 · Experiments and pilots: is the difference real or luck?
AI Evals in Practice · Author: Shivank Agrawal

Pilot: 500 students used the new DoubtBot (B), 500 used the old one (A).
Resolution rate (doubt solved without a human): B = 352/500, A = 322/500.
The two-proportion z-test asks: how surprising is a gap this big if the versions were equal?

No API key needed for this one.
Run:  python ch4_01_pilot_ab_test.py
"""
import math
from statistics import NormalDist


def two_proportion_test(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)                               # overall rate if there were no difference
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))    # expected noise in the gap
    z = (p1 - p2) / se                                           # how many "noise units" the gap is
    p_value = 2 * (1 - NormalDist().cdf(abs(z)))                 # chance of a gap this big by luck alone
    return round(p1 - p2, 3), round(z, 2), round(p_value, 3)


gap, z, p = two_proportion_test(352, 500, 322, 500)
print(f"B - A = {gap:+.1%}   z = {z}   p-value = {p}")
print("Real improvement (p < 0.05)" if p < 0.05 else "Could be luck: run longer or with more students")

# Same gap, smaller pilot: the evidence is much weaker
gap, z, p = two_proportion_test(70, 100, 64, 100)
print(f"\nSmall pilot: B - A = {gap:+.1%}   z = {z}   p-value = {p}")
print("Real improvement (p < 0.05)" if p < 0.05 else "Could be luck: run longer or with more students")

# Output:
# B - A = +6.0%   z = 2.02   p-value = 0.043
# Real improvement (p < 0.05)
#
# Small pilot: B - A = +6.0%   z = 0.9   p-value = 0.367
# Could be luck: run longer or with more students
