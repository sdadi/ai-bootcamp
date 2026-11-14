"""
Chapter 5.5 · Multimodal: word error rate (WER) for a voice mode
AI Evals in Practice · Author: Shivank Agrawal

If DoubtBot gets a voice mode, how accurate is its speech-to-text?
WER = (substitutions + insertions + deletions) / number of words in the reference.
We find the smallest number of word edits with the classic edit-distance table.

No API key needed for this one.
Run:  python ch5_04_word_error_rate.py
"""


def wer(reference, transcript):
    ref, hyp = reference.lower().split(), transcript.lower().split()
    # cost[i][j] = (edits, subs, ins, dels) to turn ref[:i] into hyp[:j]
    cost = [[None] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        cost[i][0] = (i, 0, 0, i)                       # delete all i reference words
    for j in range(len(hyp) + 1):
        cost[0][j] = (j, 0, j, 0)                       # insert all j transcript words
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost[i][j] = cost[i - 1][j - 1]         # same word: no edit
                continue
            diag, left, up = cost[i - 1][j - 1], cost[i][j - 1], cost[i - 1][j]
            cost[i][j] = min(                           # cheapest way wins
                (diag[0] + 1, diag[1] + 1, diag[2], diag[3]),   # substitute one word
                (left[0] + 1, left[1], left[2] + 1, left[3]),   # insert an extra word
                (up[0] + 1, up[1], up[2], up[3] + 1))           # delete a missing word
    edits, subs, ins, dels = cost[-1][-1]
    return {"S": subs, "I": ins, "D": dels, "WER": round(edits / len(ref), 2)}


tests = [
    ("why is my heap push slow", "why is my heap bush so slow"),      # 1 substitution + 1 insertion
    ("what is the time complexity of heapify", "what is time complexity of heapify"),  # 1 deletion
    ("explain binary search", "explain binary search"),               # perfect
]
for ref, hyp in tests:
    print(wer(ref, hyp), f"| said: {ref!r}  heard: {hyp!r}")

# Output:
# {'S': 1, 'I': 1, 'D': 0, 'WER': 0.33} | said: 'why is my heap push slow'  heard: 'why is my heap bush so slow'
# {'S': 0, 'I': 0, 'D': 1, 'WER': 0.14} | said: 'what is the time complexity of heapify'  heard: 'what is time complexity of heapify'
# {'S': 0, 'I': 0, 'D': 0, 'WER': 0.0} | said: 'explain binary search'  heard: 'explain binary search'
#
# Note: WER treats every word the same. "push" -> "bush" changes the meaning of a DSA doubt;
# a dropped "the" does not. That is why multimodal evals still need human review.
