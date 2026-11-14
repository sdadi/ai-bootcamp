"""
Chapter 5.6 · Measuring "sounds human": an implied-meaning judge
AI Evals in Practice · Author: Shivank Agrawal

Linguists call it pragmatics: what a person MEANS beyond the literal words.
"guess I'm just not a coding person lol" is not a request for a quiz question.
We turn that idea into a 1-5 rubric with every level defined, and judge two replies.

Run:  python ch5_05_implied_meaning_judge.py
"""
from pydantic import BaseModel

from judges import judge                    # the Chapter 3 judge() function

LEVELS_TABLE = """
5 = names the implied feeling or need, responds to it warmly, and adapts the help to it
4 = clearly acknowledges the implied meaning and adapts the help, but a little generically
3 = acknowledges it in passing, then continues mostly as if it was not said
2 = a generic comforting line, then ignores it completely
1 = ignores or misreads the implied meaning entirely"""

IMPLIED = """You grade ONE thing: does the tutor understand what the student MEANS beyond the literal words?
Implied meaning = feelings, intentions or needs expressed indirectly (jokes, 'lol', understatement, incomplete sentences).
Steps: 1) Write the student's likely implied meaning. 2) Check whether the reply acknowledges it.
3) Check whether tone and content change because of it. Then pick a score using these levels: {levels}
Return JSON: {{"implied_meaning": str, "reason": str, "score": 1-5}}"""


class ImpliedMeaning(BaseModel):
    implied_meaning: str
    reason: str
    score: int


student = "got 3/10 in the heaps quiz again. guess I'm just not a coding person lol"
replies = {
    "quick pivot": "Quiz scores don't define you! Let's practise: what is the time complexity of heappush?",
    "reads the room": ("Getting 3/10 twice is really frustrating, and it makes sense to doubt yourself. "
                       "But heaps trip up almost everyone at first; it's a topic, not a talent. "
                       "Want to go through the 2 questions you found hardest, slowly, one step at a time?"),
}

for name, reply in replies.items():
    print(f"{name}:", judge(IMPLIED.format(levels=LEVELS_TABLE), ImpliedMeaning,
                            student=student, reply=reply), "\n")

# Example output:
# quick pivot: {'implied_meaning': 'The student feels discouraged and is doubting whether they can succeed; the "lol" softens real frustration.',
#               'reason': 'The reply notices low confidence only generically and immediately pivots to a quiz question.', 'score': 2}
#
# reads the room: {'implied_meaning': 'The student is discouraged and starting to believe they are not capable of coding.',
#                  'reason': 'Names the frustration, reframes the self-doubt and offers slower, targeted help.', 'score': 5}
