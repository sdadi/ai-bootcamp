"""
Chapter 4.5 · Monitoring DoubtBot: judging a sample of live traffic every night
AI Evals in Practice · Author: Shivank Agrawal

Judging every production reply is expensive, so each night we:
  1. take a RANDOM slice of the day's traces and run the integrity judge on it
     -> an unbiased estimate of the failure rate (only random samples give unbiased rates)
  2. send ALL flagged traces (thumbs-down or escalated) to TAs for a human look
     -> great for finding new failure types, but NOT for estimating rates (they are biased)
  3. alert if the estimated failure rate crosses the threshold

data/live_traces.json = yesterday's (anonymised) production traces.

Run:  python ch4_02_nightly_monitoring.py
"""
import json
import random
from pathlib import Path

from judges import INTEGRITY, Verdict, judge


def alert(message):                          # in production: Slack / PagerDuty / email
    print("🚨 ALERT:", message)


def queue_for_ta_review(traces):             # in production: a review queue in your labelling tool
    for t in traces:
        print(f"   review queue <- {t['id']}  thumbs_down={t['thumbs_down']}  escalated={t['escalated']}")


def nightly_quality_check(todays_traces, rate=0.5, threshold=0.02, seed=7):
    rng = random.Random(seed)                                                    # fixed seed: same demo every run
    random_part = [t for t in todays_traces if rng.random() < rate]              # unbiased random slice
    flagged = [t for t in todays_traces if t["thumbs_down"] or t["escalated"]]   # look at ALL of these

    fails = sum(judge(INTEGRITY, Verdict, question=t["question"], reply=t["answer"])["verdict"] == "FAIL"
                for t in random_part)
    rate_est = fails / max(len(random_part), 1)              # estimate from the RANDOM slice only
    if rate_est > threshold:
        alert(f"Integrity failures ~{rate_est:.1%} today (threshold {threshold:.0%})")
    queue_for_ta_review(flagged)                             # humans read the flagged ones
    return {"sampled": len(random_part), "fails": fails,
            "fail_rate_est": round(rate_est, 3), "flagged_for_review": len(flagged)}


traces = json.loads((Path(__file__).parent / "data" / "live_traces.json").read_text())
print(f"{len(traces)} traces yesterday")
print(nightly_quality_check(traces))

# Example output:
# 19 traces yesterday
# 🚨 ALERT: Integrity failures ~8.3% today (threshold 2%)
#    review queue <- P005  thumbs_down=True  escalated=False
#    review queue <- P009  thumbs_down=True  escalated=False
#    review queue <- P017  thumbs_down=False  escalated=True
#    review queue <- P019  thumbs_down=False  escalated=True
# {'sampled': 12, 'fails': 1, 'fail_rate_est': 0.083, 'flagged_for_review': 4}
#
# The random slice caught P011 (full LIS code). P005 (full linked-list code) was not sampled,
# but its thumbs-down sent it to the review queue anyway: that's why you use BOTH routes.
# In production the slice is ~5% of thousands of traces. Here we sample 50% of 19 so the demo shows something.
