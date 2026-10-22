# Vendored from StartLux-Decision v1.1's package (startlux_decide/loops.py, StartLuxLabs/Startlux-decide-inference
# commit 78e39fc), Apache-2.0; see LICENSES/StartLux-Decision-Apache-2.0.txt.  Unchanged: where a thought is cut is part
# of what the read-out after it sees, so StartLux-Prism must cut exactly where the reference does (tests/test_think.py
# checks both on many synthetic texts).
"""Loop detector for thoughts (the release uses this rule set, "version 3").

A decision model that has reached its answer sometimes cannot close its thought: it repeats the answer letter, a value,
a sentence or a whole paragraph until the token budget is used. The thought is generated in chunks, and between chunks
``looping(text)`` tells whether the END of the text is such a loop; ``loop_cut(text)`` returns the length of the
shortest prefix that already ends in the loop, so the reasoning and the first repeats are kept and the rest is dropped.

Rules, applied to the tail of the text:

* short   one unit of at most 80 characters repeated at least max(4, ceil(60 / unit length)) times (window of 400
          characters; every run of digits replaced by "#", checked with and without whitespace normalised)
* digits  the last 150 characters are at least 90% digits (a number that never ends)
* long    one unit of 81 to 400 characters repeated at least 3 times (window of 1,600 characters; digits replaced,
          whitespace normalised)
* lines   the last 20 non-empty lines are at most 8 different lines (spacing normalised, digits kept)
* words   the last 60 words are at most 6 different words (digits replaced)

At sampling temperature 0.6 many endless thoughts are not periodic: the model shuffles a handful of lines in random
order. The line and word rules catch those. The line rule keeps digits, so lines that differ only in their numbers
count as different lines; the periodic rules replace digits, so rows that repeat one pattern with changing numbers
are still a loop for them.
"""
from __future__ import annotations

import re

_DIG, _WS = re.compile(r"\d+"), re.compile(r"\s+")

SHORT_WINDOW, SHORT_MAX_UNIT, SHORT_MIN_REPEATS, SHORT_MIN_SPAN = 400, 80, 4, 60
DIGIT_RUN = 150
LONG_WINDOW, LONG_MAX_UNIT, LONG_MIN_REPEATS = 1600, 400, 3
LINES, MAX_DISTINCT_LINES = 20, 8
WORDS, MAX_DISTINCT_WORDS = 60, 6


def _periodic(t: str, lo: int, hi: int, min_rep: int, min_span: int) -> bool:
    """t ends in a unit of lo..hi characters repeated at least max(min_rep, ceil(min_span / unit)) times."""
    n = len(t)
    for p in range(lo, hi + 1):
        r = max(min_rep, -(-min_span // p))
        if n >= r * p and all(t[n - (j + 1) * p:n - j * p] == t[n - p:] for j in range(1, r)):
            return True
    return False


def few_lines(text: str) -> bool:
    """The last LINES non-empty lines are at most MAX_DISTINCT_LINES different lines (spacing normalised, digits
    kept)."""
    ls = [x for x in (_WS.sub(" ", y).strip() for y in text[-80 * LINES:].split("\n")) if x]
    return len(ls) > LINES and len(set(ls[-LINES:])) <= MAX_DISTINCT_LINES   # > LINES: the first line may be cut


def low_diversity(text: str) -> bool:
    """The last WORDS words are at most MAX_DISTINCT_WORDS different words (digits normalised)."""
    w = _DIG.sub("#", text[-12 * WORDS:]).split()
    return len(w) > WORDS and len(set(w[-WORDS:])) <= MAX_DISTINCT_WORDS


def looping(text: str) -> bool:
    """True when the text ends in a loop (see the module docstring)."""
    u = _DIG.sub("#", text[-SHORT_WINDOW:])
    for t in (u, _WS.sub(" ", u)):
        if _periodic(t, 1, SHORT_MAX_UNIT, SHORT_MIN_REPEATS, SHORT_MIN_SPAN):
            return True
    d = text[-DIGIT_RUN:]
    if len(d) == DIGIT_RUN and sum(c.isdigit() for c in d) >= 0.9 * DIGIT_RUN:
        return True
    v = _WS.sub(" ", _DIG.sub("#", text[-LONG_WINDOW:]))
    if _periodic(v, SHORT_MAX_UNIT + 1, LONG_MAX_UNIT, LONG_MIN_REPEATS, 0):
        return True
    return few_lines(text) or low_diversity(text)


def loop_cut(text: str) -> int | None:
    """Length of the shortest prefix that already ends in the loop, or None when the text does not end in one.
    Found by bisection to within 8 characters: inside a loop every longer prefix also ends in the loop."""
    if not looping(text):
        return None
    lo, hi = 0, len(text)
    while hi - lo > 8:
        mid = (lo + hi) // 2
        if looping(text[:mid]):
            hi = mid
        else:
            lo = mid
    return hi
