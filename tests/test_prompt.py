"""Prompt rendering without jinja equals the checkpoint's chat template, byte for byte and token for token.

    python tests/test_prompt.py MODEL_DIR JEVBENCH_PUBLIC_DIR

Runs on any machine with `tokenizers` and `transformers` (no GPU, no torch).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flash_jev import jevfmt as J  # noqa: E402
from flash_jev.prompt import Tokenizer, common_prefix, render_text, rows_for  # noqa: E402

EXTRA = [
    ({"ticket": "I was charged twice for order #4411 and the app still shows it as unpaid."},
     {"team": {"type": "choice", "instructions": "Which team should handle this ticket?",
               "criteria": {"billing": "Payments, refunds and invoices", "shipping": "Delivery and tracking",
                            "technical": "App, login and account problems"}},
      "urgent": {"type": "noul", "instructions": "Should this ticket be answered today?"},
      "severity": {"type": "score", "instructions": "How severe is the impact?",
                   "criteria": ["cosmetic", "annoying", "blocks the customer"]}}),
    ("", {"q": {"type": "noul", "instructions": "Empty state?"}}),
    ("ends with letters", {"q": {"type": "noul", "instructions": "x"}}),
    ("ends with space ", {"q": {"type": "noul", "instructions": "x"}}),
    ("ends with newlines\n\n", {"q": {"type": "noul", "instructions": "Question: nested\n\nQuestion: twice"}}),
    ("trailing punctuation }])!?\n", {"q": {"type": "noul", "instructions": "x"}}),
    ("Ünïcödé — 中文 テスト 🚀\n\tTabs and\r\nnewlines", {"q": {"type": "choice", "instructions": "Pick one",
                                                          "criteria": {"A": "first", "B": "second"}}}),
    ([1, 2, 3, {"a": None}] * 3, {"q": {"type": "score", "instructions": "How much?",
                                        "criteria": {"0": "low", "1": "mid", "2": "high"}}}),
]


def main():
    model_dir, bench = sys.argv[1], sys.argv[2]
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(model_dir)
    tok = Tokenizer(model_dir)
    cases = list(EXTRA)
    for tier in ("easy", "original", "hard"):
        for line in open(os.path.join(bench, f"{tier}.jsonl")):
            raw = json.loads(line)
            cases.append((raw["state"], {"decision": raw["question"]}))
    n = 0
    for state, questions in cases:
        recs = [J.from_systemone(state, q) for q in questions.values()]
        rows = rows_for(tok, recs, list(questions))
        for rec, row in zip(recs, rows):
            msgs, _ = J.messages(rec)
            want = hf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            got, _ = render_text(rec)
            assert got == want, (got, want)
            assert row.ids == hf.encode(want, add_special_tokens=False), "split tokenization differs"
            assert row.ids == tok.encode(got)
            assert row.ids[:len(tok.system_ids)] == tok.system_ids
            n += 1
        if len(rows) > 1:
            c = common_prefix([r.ids for r in rows])
            assert c >= len(tok.system_ids)
    assert tok.letters == J.check_tokenizer(hf)
    # the head and tail caches under concurrent requests (small caches, so they evict all the time)
    import threading
    errors = []

    def worker(k):
        try:
            for i in range(300):
                text = f"{k} {i} evidence " * (i % 7 + 1)
                assert tok.encode_head(text, cache_size=16) == tok.encode(text)
                tails = [f"Question: {k} {j} {i % 5}?" for j in range(3)]    # repeats across threads and turns
                assert tok.encode_tails(tails, cache_size=8) == [tok.encode(t) for t in tails]
        except Exception as e:                          # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:3]
    print(f"ok: {n} prompts identical to the chat template; system prefix {len(tok.system_ids)} tokens; head and "
          f"tail caches thread-safe")


if __name__ == "__main__":
    main()
