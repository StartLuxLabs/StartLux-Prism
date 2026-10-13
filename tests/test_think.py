"""AutoThink against StartLux-Decision v1.1's own package (startlux_decide), on the CPU.

    STARTLUX_DECIDE=/path/to/Startlux-decide-inference python tests/test_think.py --v11-dir V11_MODEL_DIR \\
        [--n 300] [--texts 5000] [--seed 0]

1. loop detector: flash_jev.loops against startlux_decide.loops (looping and loop_cut) on --texts synthetic texts:
   periodic units of 1-400 characters with digits that change, digit runs, shuffled lines, word salads, prose, with
   random prefixes, cuts and whitespace.
2. thinking prompts: think.Framing (the folder's chat template with thinking on, rendered once) against
   startlux_decide's think_prompt (transformers' apply_chat_template) on random questions: the same text byte for byte
   and the same tokens; and the read prompts after a thought tokenized the same.
3. orchestration: whole requests (choice, yes/no, score, knock-out lists, none_check / allow_none, think_mode auto and
   always, gates, mixes, budgets, stream and chunked) through flash-jev's ThinkingDecider and through startlux_decide's
   Decider, both on deterministic fake engines that read the same pseudo-random letter logits for the same prompt text
   and write the same thoughts - token ids from a hash of the thinking prompt and the seed: closing the thought,
   ending with the end-of-sequence token, running into the budget or into a loop, a stop string split over tokens,
   multi-byte characters, stray think tags.  flash-jev's side runs its scheduler path (think.StreamThought /
   ChunkGeneration fed one token at a time, its incremental detokenizer); the package's side gets vLLM 0.31's stream
   events built independently here (the generated ids decoded whole at every step).  Answers (every float), details
   (thought text, tokens, how it ended, both read-outs) and usage must be identical.
"""
import argparse
import asyncio
import copy
import hashlib
import os
import random
import sys
import time
from concurrent.futures import Future

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from flash_jev import jevfmt as J  # noqa: E402
from flash_jev import loops as FL  # noqa: E402
from flash_jev import think as TH  # noqa: E402
from flash_jev.engine.decode import EOS  # noqa: E402
from flash_jev.prompt import Tokenizer  # noqa: E402
import test_v11 as V  # noqa: E402

STOP = "</think>"
CONTEXT = 65536


# ---------------------------------------------------------------------------------------------------- 1. loops
WORDS = ("the evidence option fits because so is likely not answer billing refund charged twice customer team order "
         "shipping B A C D yes no score level 3 12 2024 $1,140 INV-2291").split()


def synthetic_text(r):
    """A text that may end in a loop of one of the detector's kinds."""
    k = r.randrange(12)
    pre = " ".join(r.choice(WORDS) for _ in range(r.randrange(0, 200)))
    pre += r.choice(["", "\n", "\n\n", " ", ". "])
    if k == 0:                                           # short unit
        unit = " ".join(r.choice(WORDS) for _ in range(r.randrange(1, 6))) + r.choice([" ", ". ", "\n", ""])
        body = unit * r.randrange(1, 40)
    elif k == 1:                                         # long unit
        unit = " ".join(r.choice(WORDS) for _ in range(r.randrange(12, 70))) + r.choice([".\n", " ", "\n\n"])
        body = unit * r.randrange(1, 6)
    elif k == 2:                                         # rows of one pattern, changing numbers
        body = "".join(f"Order #{4000 + i}: {r.randrange(9)} items, total ${r.randrange(999)}.\n"
                       for i in range(r.randrange(1, 40)))
    elif k == 3:                                         # a number that never ends
        body = "".join(r.choice("0123456789") for _ in range(r.randrange(50, 400)))
        if r.random() < 0.3:
            body = body[:len(body) // 2] + r.choice(WORDS) + body[len(body) // 2:]
    elif k == 4:                                         # shuffled lines
        lines = [" ".join(r.choice(WORDS) for _ in range(r.randrange(2, 9))) for _ in range(r.randrange(2, 14))]
        body = "\n".join(r.choice(lines) for _ in range(r.randrange(5, 50))) + "\n"
    elif k == 5:                                         # word salad
        vocab = r.sample(WORDS, r.randrange(3, 10))
        body = " ".join(r.choice(vocab) for _ in range(r.randrange(20, 140)))
    elif k == 6:                                         # spacing varies
        unit = r.choice(WORDS) + r.choice([" ", "  ", "\n", " \n"])
        body = "".join(unit.replace(" ", " " * r.randrange(1, 3)) for _ in range(r.randrange(5, 80)))
    elif k == 7:                                         # letters
        body = "".join(r.choice(["B", "B.", "B\n", "A", "Answer: B\n"]) for _ in range(r.randrange(5, 120)))
    elif k == 8:                                         # loop, then a new start
        unit = " ".join(r.choice(WORDS) for _ in range(r.randrange(1, 5))) + " "
        body = unit * r.randrange(5, 30) + " ".join(r.choice(WORDS) for _ in range(r.randrange(1, 30)))
    else:                                                # prose
        body = " ".join(r.choice(WORDS) for _ in range(r.randrange(10, 600)))
    t = pre + body
    if r.random() < 0.3:
        t = t[:r.randrange(len(t) + 1)]
    return t


def check_loops(n, seed):
    from startlux_decide import loops as RL
    r = random.Random(seed)
    bad = loops = cuts = 0
    for i in range(n):
        t = synthetic_text(r)
        a, b = FL.looping(t), RL.looping(t)
        ca, cb = FL.loop_cut(t), RL.loop_cut(t)
        loops += a
        cuts += ca is not None and ca < len(t)
        if a != b or ca != cb:
            bad += 1
            if bad < 5:
                print("loop detector differs:", repr(t[-200:]), a, b, ca, cb)
    print(f"1. loop detector: {n} texts ({loops} end in a loop, {cuts} cut before their end): "
          f"{'looping and loop_cut identical to startlux_decide' if not bad else f'{bad} DIFFER'}", flush=True)
    return bad


# ---------------------------------------------------------------------------------------------------- 2. prompts
def check_prompts(n, seed, framing, ftok, htok):
    from startlux_decide import prompt as P
    r = random.Random(seed + 1)
    bad = count = 0
    for _ in range(n):
        st, q = V.state(r), V.question(r)
        none = r.random() < 0.5 and q["type"] == "choice"
        try:
            rec = J.from_systemone(st, q, allow_none=none)
            ours = framing.prompt(rec)
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
        theirs = P.think_prompt(htok, P.messages(P.from_systemone(st, q, none)))
        count += 1
        thought = r.choice(["", "B fits.", "The refund rule applies.\nB fits.", "x <think> y", "€ 25 → 30 ✓ 日本",
                            "a" * r.randrange(5000)])
        read = TH.read_prompt(TH.AutoThink(), ours, thought, r.choice(["closed", "loop", "budget"]))[1]
        if ours != theirs:
            bad += 1
            if bad < 4:
                print("thinking prompt differs:\n", repr(ours[:300]), "\n", repr(theirs[:300]))
        elif (ftok.encode(ours + TH.RELEASE_PREFIX) != htok(ours + TH.RELEASE_PREFIX, add_special_tokens=False)["input_ids"]
              or ftok.encode(read) != htok(read, add_special_tokens=False)["input_ids"]):
            bad += 1
            print("tokens differ:", repr(ours[-200:]))
    print(f"2. thinking prompts: {count} rendered, {'all identical (text and tokens, read prompts too)' if not bad else f'{bad} DIFFER'}",
          flush=True)
    return bad


# ---------------------------------------------------------------------------------------------------- 3. orchestration
SNIPPETS = ["The evidence says the customer was charged twice. ", "So option B fits best.\n",
            "Let me check the options again.\n", "Option A covers refunds; option C covers outages.\n",
            "Total: $1,140 on INV-2291 and INV-2292. ", "€ 25 → 30 ✓ 日本語のテキスト。", "😀 emoji ", "<think> stray ",
            "Hmm, wait. ", "B. ", "1234567890", "\n\n", "None of the above may apply. "]


def hrng(*parts):
    return random.Random(hashlib.sha256("\x00".join(map(str, parts)).encode("utf-8", "surrogatepass")).digest())


def fake_thought(ftok, prompt, seed, max_tokens):
    """Token ids of a deterministic thought for (prompt text, seed): the same on both sides."""
    r = hrng("thought", prompt, seed)
    kind = r.choice(["close", "close", "eos", "budget", "loop", "loop_lines", "digits", "split_stop", "close_late"])
    text = "".join(r.choice(SNIPPETS) for _ in range(r.randrange(1, 40)))
    enc = ftok.encode
    if kind in ("close", "close_late"):
        if kind == "close_late":
            text += "".join(r.choice(SNIPPETS) for _ in range(r.randrange(100, 400)))
        return enc(text + "</think>\n\nB")
    if kind == "eos":
        return enc(text) + [r.choice(EOS)]
    if kind == "split_stop":                           # "</think>" from three tokens: the text-level stop check
        return enc(text) + enc("</") + enc("think") + enc(">") + enc(" after")
    if kind == "loop":
        unit = r.choice(SNIPPETS[:5] + ["B. ", "The answer is B.\n"])
        ids = enc(text + unit * 1200)
    elif kind == "loop_lines":
        lines = [r.choice(SNIPPETS[:5]).strip() for _ in range(4)]
        ids = enc(text + "\n".join(r.choice(lines) for _ in range(700)))
    elif kind == "digits":
        ids = enc(text + "".join(r.choice("0123456789") for _ in range(9000)))
    else:
        ids = enc(text)
    while len(ids) < max_tokens + 8:                   # runs into the budget
        ids += enc(" " + " ".join(r.choice(WORDS) for _ in range(200)))
    return ids


def vllm_events(htok, ids, max_tokens):
    """vLLM 0.31's OpenAI stream for these sampled ids, built independently of flash-jev's emulation: the generated
    ids decoded whole at every step (an incomplete character at the end waits), stop string "</think>" cut from the
    text, the end-of-sequence token not decoded, len(stop) - 1 characters held back until the last event.
    -> [(text, tokens, finish or None)] as startlux_decide's VLLMBackend.stream_at yields them (no empty event)."""
    out, gen, sent = [], [], 0
    hold = len(STOP) - 1
    for k, t in enumerate(ids[:max_tokens], 1):
        if t in EOS:
            out.append((text_now(htok, gen)[sent:], k, "stop"))
            return out
        gen.append(t)
        full = text_now(htok, gen)
        i = full.find(STOP)
        if i >= 0:
            out.append((full[:i][sent:], k, "stop"))
            return out
        if k == max_tokens:
            out.append((full[sent:], k, "length"))
            return out
        end = len(full) - hold
        if end > sent:
            out.append((full[sent:end], k, None))
            sent = end
    raise AssertionError("a fake thought ran out of tokens")


def text_now(htok, gen):
    return htok.decode(gen, skip_special_tokens=True).rstrip("�")


def logits_for(prompt, k):
    rr = random.Random(int(hashlib.sha1(prompt.encode("utf-8", "surrogatepass")).hexdigest()[:16], 16))
    return [rr.gauss(0.0, 2.0) for _ in range(k)]


class PkgEngine:
    """startlux_decide backend: read-outs and thoughts of the fake engine."""

    def __init__(self, htok, ftok):
        from startlux_decide.backends.base import Backend, Generation, StreamEvent
        outer = self

        class _B(Backend):
            name = "fake-vllm"
            streams = True
            context_length = CONTEXT
            tokenizer = htok

            async def readout(self, prompt, letter_ids):
                return logits_for(prompt, len(letter_ids))

            async def generate(self, prompt, max_tokens, sampling, stop, seed):
                ev = vllm_events(htok, fake_thought(ftok, prompt, seed, max_tokens), max_tokens)
                return Generation("".join(e[0] for e in ev), ev[-1][1], "stop" if ev[-1][2] == "stop" else "length")

            async def generate_stream(self, prompt, max_tokens, sampling, stop, seed):
                outer.seeds.append(seed)
                for text, tokens, finish in vllm_events(htok, fake_thought(ftok, prompt, seed, max_tokens), max_tokens):
                    yield StreamEvent(text, tokens, finish)
        self.backend = _B()
        self.seeds = []


class FjBackend:
    """flash-jev side: letter logits of the same fake engine (the prompt text decoded back from its tokens)."""

    def __init__(self, ftok):
        self.ftok = ftok

    def letter_logits(self, rows):
        return [logits_for(self.ftok.tok.decode(r.ids, skip_special_tokens=False), r.n_options) for r in rows]


class FjSched:
    """flash-jev side: the scheduler's interface, the fake engine's tokens fed through each job's consumer as
    ThinkScheduler feeds sampled tokens."""

    def __init__(self, ftok):
        self.ftok = ftok
        self.seeds = []

    def think(self, job):
        prompt = self.ftok.tok.decode(job.ids, skip_special_tokens=False)
        self.seeds.append(job.seed)
        for t in fake_thought(self.ftok, prompt, job.seed, job.max_tokens):
            if job.consumer.add(t):
                break
        f = Future()
        f.set_result((job.consumer.result(), job.t0, time.perf_counter()))
        return f

    def submit_async(self, rows):
        f = Future()
        f.set_result(([logits_for(self.ftok.tok.decode(r, skip_special_tokens=False), 26) for r in rows], {}))
        return f


def strip_ms(x):
    if isinstance(x, dict):
        return {k: strip_ms(v) for k, v in x.items() if k not in ("ms", "total_ms")}
    if isinstance(x, (list, tuple)):
        return [strip_ms(v) for v in x]
    return x


def check_orchestration(n, seed, model_dir, framing, ftok, htok):
    from startlux_decide.controls import Controls as PControls
    from startlux_decide.modelcfg import load_model_folder
    from startlux_decide.pipeline import Decider as PDecider
    folder = load_model_folder(model_dir)
    auto = TH.folder_auto_think(model_dir)
    pe = PkgEngine(htok, ftok)
    r = random.Random(seed + 2)
    bad = n_req = n_thought = 0
    ends = {}
    for i in range(n):
        st = V.state(r)
        qs = {f"q{j}": (V.wide_question(r) if r.random() < 0.06 else V.question(r)) for j in range(r.randrange(1, 4))}
        kw = dict(think_mode=r.choice(["auto", "auto", "always"]), gate=r.choice([0.9, 0.9, 0.6, 1.0]),
                  budget=r.choice([4096, 4096, 1500, 300, 40]), mix=r.choice([0.75, 0.75, 0.5, 1.0, 0.0]),
                  details=True, thought_text=True, thought_generation=r.choice(["stream", "stream", "chunked"]),
                  allow_none=r.random() < 0.3, none_check=r.random() < 0.7)
        pctl = PControls(**kw)
        fctl = TH.Controls(**kw)
        try:
            theirs = asyncio.run(PDecider(folder, pe.backend).decide(st, copy.deepcopy(qs), pctl))
            theirs = (theirs.answers, theirs.usage)
        except ValueError as e:
            theirs = ("refused", type(e).__name__)
        fs = FjSched(ftok)
        dec = TH.ThinkingDecider(ftok, FjBackend(ftok), fs, framing, auto, CONTEXT)
        try:
            a, u, _ = dec.decide(st, copy.deepcopy(qs), ctl=fctl)
            ours = (a, u)
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            ours = ("refused", type(e).__name__)
        n_req += 1
        if isinstance(ours[0], str) or isinstance(theirs[0], str):
            if isinstance(ours[0], str) != isinstance(theirs[0], str):
                bad += 1
                print("refusal differs:", qs, ours, theirs)
            continue
        for ans in theirs[0].values():
            th = ans["details"]["thought"]
            n_thought += th["used"]
            if th["used"]:
                ends[th["ended"]] = ends.get(th["ended"], 0) + 1
        if strip_ms(ours) != strip_ms(theirs):
            bad += 1
            if bad < 4:
                print("request differs:", kw, qs, "\n ours  ", strip_ms(ours), "\n theirs", strip_ms(theirs))
    print(f"3. orchestration: {n_req} requests, {n_thought} thoughts (ended: {ends}): "
          f"{'answers, details and usage identical to startlux_decide' if not bad else f'{bad} DIFFER'}", flush=True)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v11-dir", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--texts", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, os.environ["STARTLUX_DECIDE"])
    from transformers import AutoTokenizer
    htok = AutoTokenizer.from_pretrained(a.v11_dir)
    ftok = Tokenizer(a.v11_dir)
    framing = TH.Framing(a.v11_dir)
    bad = check_loops(a.texts, a.seed)
    bad += check_prompts(a.n * 10, a.seed, framing, ftok, htok)
    bad += check_orchestration(a.n, a.seed, a.v11_dir, framing, ftok, htok)
    print("all ok" if not bad else f"{bad} FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
