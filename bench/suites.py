"""Fixed request suites, shared by every engine under test so numbers are comparable across engines and machines.

- startlux3 / startlux1: StartLux-Decision's eval/latency.py requests (one choice + one yes/no + one score on a support
  ticket, 891 prompt tokens over the three questions; and its single yes/no).  StartLux publishes 26.0 / 14.7 ms for
  the 4B on one H200 over HTTP with these (docs/inference.md), and quotes other engines on the same shape.
- readme3, yesno, longpage2, wide33, review1: StartLuxLabs' StartLux-Decision-inference sample requests (sl_mlx.py):
  DEMO (the StartLux-Decision README request, also its server's start-up demo), YESNO (DEMO's yes/no alone), and
  requests() - its MLX-vs-torch agreement checks.  s-gen-mlx uses the same five (Apple M5 Max numbers exist).
None of the seven has graded answers: they measure speed; accuracy is JevBench's.
- jevbench: the 231 public JevBench items, one question each (accuracy and a realistic length mix).
"""
import json
import os

TICKET = {"ticket": {"id": "T-48213", "channel": "email", "customer_tier": "business",
                     "subject": "Charged twice for renewal, dashboard still shows the account as suspended",
                     "body": ("Hello, our annual renewal for the Business plan went through on the 3rd, and the card "
                              "statement shows two identical charges of 1,140 USD from you on the same day. Since then "
                              "the admin dashboard still says the workspace is suspended for non-payment, so eleven "
                              "people on my team cannot log in to the reporting module and we have a board meeting "
                              "tomorrow morning. I already tried logging out and clearing the cache, and the status page "
                              "says all systems are operational. Please refund the duplicate charge and restore access as "
                              "soon as possible. Invoice numbers INV-2291 and INV-2292 are attached.")},
          "account": {"plan": "Business annual", "seats": 11, "open_incidents": 0, "previous_tickets_90d": 1}}
TICKET_QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team should own this ticket?",
             "criteria": {"billing": "Payments, refunds and invoices", "access": "Login, suspension and permissions",
                          "technical": "Bugs and outages", "sales": "Plans and upgrades"}},
    "urgent": {"type": "noul", "instructions": "Does this ticket need a reply within one hour?"},
    "severity": {"type": "score", "instructions": "How severe is the customer impact?",
                 "criteria": ["cosmetic", "minor inconvenience", "blocks part of the team", "blocks the whole business"]}}

README_STATE = {"ticket": "I was charged twice for order #4411 and the app still shows it as unpaid."}
README_QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
             "criteria": {"billing": "Payments, refunds and invoices", "shipping": "Delivery and tracking",
                          "technical": "App, login and account problems"}},
    "urgent": {"type": "noul", "instructions": "Should this ticket be answered today?"},
    "severity": {"type": "score", "instructions": "How severe is the impact?",
                 "criteria": ["cosmetic", "annoying", "blocks the customer"]}}


def _long_page():
    state = {"page": "Order history. " + " ".join(
        f"Order #{4000 + i}: {i % 5 + 1} items, total ${17 * i % 300}.{i % 100:02d}, "
        f"status {'shipped' if i % 3 else 'pending'}." for i in range(40)),
        "user": "Where is my most recent pending order?"}
    questions = {"next": {"type": "choice", "instructions": "Which order is the user asking about?",
                          "criteria": {f"o{4000 + i}": f"Order #{4000 + i}" for i in range(0, 40, 3)}},
                 "done": {"type": "noul", "instructions": "Can the question be answered from this page alone?"}}
    return state, questions


WIDE = {"city": {"type": "choice", "instructions": "Which city is the capital of Australia?",
                 "criteria": {c.lower().replace(" ", "_"): c for c in (
                     "Sydney Melbourne Brisbane Perth Adelaide Canberra Hobart Darwin Auckland Wellington Suva "
                     "Jakarta Manila Singapore Bangkok Hanoi Tokyo Osaka Seoul Beijing Shanghai Taipei Delhi "
                     "Mumbai Dhaka Karachi Tehran Dubai Doha Riyadh Cairo Nairobi Lagos").split()}}}
STARS = {"stars": {"type": "score", "instructions": "How satisfied is the customer?", "criteria": ["1", "2", "3", "4", "5"]}}


def fixed():
    """name -> (state, questions)"""
    long_state, long_q = _long_page()
    return {
        "startlux3": (TICKET, TICKET_QUESTIONS),
        "startlux1": (TICKET, {"urgent": TICKET_QUESTIONS["urgent"]}),
        "readme3": (README_STATE, README_QUESTIONS),
        "yesno": (README_STATE, {"urgent": README_QUESTIONS["urgent"]}),
        "longpage2": (long_state, long_q),
        "wide33": ({"quiz": "Geography round."}, WIDE),
        "review1": ({"review": "The battery died after two days and support never answered."}, STARS),
    }


def jevbench(public_dir):
    """[(id, tier, state, questions, raw)] for the 231 public items."""
    out = []
    for tier in ("easy", "original", "hard"):
        with open(os.path.join(public_dir, f"{tier}.jsonl")) as f:
            for line in f:
                raw = json.loads(line)
                out.append((raw["id"], tier, raw["state"], {"decision": raw["question"]}, raw))
    return out
