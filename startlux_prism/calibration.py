"""SmoothQuant calibration requests: the CALIBRATION list of StartLux-Decision (startlux_decision/mlx_int8.py, commit
0e7a2e81b9c9, Apache-2.0; see LICENSES/StartLux-Decision-Apache-2.0.txt), unchanged, so int8 ranges come from the same
requests as StartLux's own int8 backend.  They contain no benchmark items."""

CALIBRATION = []            # (state, questions) pairs of the kinds of requests the models serve
for _t in ("My parcel was marked delivered but never arrived.", "The app crashes when I open settings on Android 15.",
           "I want a refund for the duplicate charge on invoice 7781.", "Can I change the shipping address after ordering?",
           "Password reset emails never arrive.", "The discount code SPRING20 is rejected at checkout."):
    CALIBRATION.append(({"ticket": _t}, {
        "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                 "criteria": {"billing": "Payments, refunds and invoices", "shipping": "Delivery and tracking",
                              "technical": "App, login and account problems"}},
        "urgent": {"type": "noul", "instructions": "Should this ticket be answered today?"},
        "severity": {"type": "score", "instructions": "How severe is the impact?",
                     "criteria": ["cosmetic", "annoying", "blocks the customer"]}}))
for _n in (5, 20, 60):
    CALIBRATION.append(({"page": " ".join(f"Item {i}: {['laptop', 'desk lamp', 'usb cable', 'office chair'][i % 4]} "
                                          f"${(i * 37) % 900}.99, {'in stock' if i % 3 else 'backorder'}." for i in range(_n)),
                         "goal": "Buy the cheapest in-stock chair."},
                        {"next": {"type": "choice", "instructions": "Which item should be clicked next?",
                                  "criteria": {f"item{i}": f"Item {i}" for i in range(min(_n, 12))}},
                         "done": {"type": "noul", "instructions": "Is the goal already met?"}}))
for _r in ("The hotel was spotless and the staff helpful.", "Food arrived cold and two items were missing.",
           "Average experience, nothing special.", "Absolutely terrible support, never again!"):
    CALIBRATION.append(({"review": _r}, {
        "stars": {"type": "score", "instructions": "How satisfied is the customer?", "criteria": ["1", "2", "3", "4", "5"]},
        "topic": {"type": "choice", "instructions": "What is the review mainly about?",
                  "criteria": {"service": "staff and support", "product": "food or room", "price": "cost",
                               "delivery": "delivery"}}}))
for _q, _o in (("Which planet is known as the Red Planet?", ["Venus", "Mars", "Jupiter", "Saturn"]),
               ("What is 17 * 3?", ["41", "51", "57", "61"]), ("Who wrote Hamlet?", ["Dickens", "Shakespeare", "Austen", "Twain"])):
    CALIBRATION.append(({"quiz": _q}, {"answer": {"type": "choice", "instructions": _q, "criteria": {o.lower(): o for o in _o}}}))
CALIBRATION.append(({"log": "\n".join(f"2026-09-30 12:{i:02d} {'ERROR' if i % 7 == 0 else 'INFO'} worker-{i % 3} step {i}"
                                      for i in range(40))},
                    {"failing": {"type": "noul", "instructions": "Is any worker failing repeatedly?"},
                     "worker": {"type": "choice", "instructions": "Which worker logs the most errors?",
                                "criteria": {"w0": "worker-0", "w1": "worker-1", "w2": "worker-2"}}}))
