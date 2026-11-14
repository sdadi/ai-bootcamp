"""
Chapter 3.10 · How big should the test set be?
AI Evals in Practice · Author: Shivank Agrawal

Four questions, four formulas (z = 1.96 for 95% confidence):
  1. Measure an average score within ± E          n = (z·σ / E)²            σ = standard deviation
  2. Measure a pass rate within ± E               n = z²·p(1−p) / E²
  3. Prove a guardrail (almost) never fails       zero failures in n cases → rate < 3/n  (rule of three)
  4. Detect a gain of Δ between two versions      n = ((z_α/2 + z_β)·σ_d / Δ)²   paired, same test cases

No API key needed for this one.
Run:  python ch3_06_test_set_size.py
"""
import math
import statistics
from statistics import NormalDist


def z(conf):                    # 0.95 → 1.96
    return NormalDist().inv_cdf(1 - (1 - conf) / 2)


# 1) average score: use the standard deviation from a small pilot run
def n_for_mean(pilot_scores, margin, conf=0.95):
    sigma = statistics.stdev(pilot_scores)          # standard deviation, NOT variance
    return math.ceil((z(conf) * sigma / margin) ** 2), round(sigma, 3)


# 2) pass rate: p_guess = what you expect; 0.5 if you have no idea (the worst case)
def n_for_pass_rate(p_guess, margin, conf=0.95):
    return math.ceil(z(conf) ** 2 * p_guess * (1 - p_guess) / margin ** 2)


# 3) rule of three: clean cases needed to claim "failure rate below max_rate" (95%)
def n_for_zero_failures(max_rate):
    return math.ceil(3 / max_rate)


# 4) paired comparison of version A vs B on the SAME cases
#    sigma_d = std dev of the per-case score differences (B − A) from a pilot
def n_for_paired_difference(sigma_d, delta, conf=0.95, power=0.80):
    z_beta = NormalDist().inv_cdf(power)            # 0.80 power → 0.84
    return math.ceil(((z(conf) + z_beta) * sigma_d / delta) ** 2)


pilot = [0.2, 0.9, 0.7, 1.0, 0.4, 0.9, 0.6, 1.0, 0.8, 0.3,
         1.0, 0.7, 0.9, 0.5, 1.0, 0.8, 0.6, 0.9, 1.0, 0.7]      # judge scores from 20 pilot doubts

print("1. mean score ± 0.05:       n, sigma =", n_for_mean(pilot, margin=0.05))
print("2. pass rate ~0.9 ± 0.05:   n =", n_for_pass_rate(0.9, margin=0.05))
print("   pass rate unknown ± 0.05: n =", n_for_pass_rate(0.5, margin=0.05))
print("3. privacy leaks < 1%:      n =", n_for_zero_failures(0.01), "clean cases")
print("   privacy leaks < 0.5%:    n =", n_for_zero_failures(0.005), "clean cases")
print("4. detect +0.05 (σ_d=0.3):  n =", n_for_paired_difference(sigma_d=0.3, delta=0.05), "paired cases")

# Output:
# 1. mean score ± 0.05:       n, sigma = (93, 0.246)
# 2. pass rate ~0.9 ± 0.05:   n = 139
#    pass rate unknown ± 0.05: n = 385
# 3. privacy leaks < 1%:      n = 300 clean cases
#    privacy leaks < 0.5%:    n = 600 clean cases
# 4. detect +0.05 (σ_d=0.3):  n = 283 paired cases
