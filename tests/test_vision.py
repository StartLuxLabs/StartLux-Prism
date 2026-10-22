"""Images in the evidence, on the CPU: what can be checked without a GPU or a reference engine.

    python tests/test_vision.py --model-dir V11_MODEL_DIR [--startlux-decide PATH]

1. requests: the images' places in the state, the resize rule and token counts, decoding and its errors
   (StartLux-Decision v1.1's cases, startlux_decide/tests/test_images.py).
2. pixels: the tower's input (vision.VisionTower.patches, run on the CPU) against transformers' Qwen2-VL image processor
   (its Pillow backend, the folder's preprocessor_config.json), bit for bit after the bf16 rounding the model applies.
3. prompts: rows with images (images.rows_for) against StartLux-Decision v1.1's own prompt (startlux_decide's
   plain_prompt over the folder's chat template, the processor's expansion of each <|image_pad|>, the folder's
   tokenizer): the same token ids; and their positions against transformers' get_rope_index for Qwen3.5 and an
   independent token-by-token walk; last_pos is the largest position.  (Needs transformers; with --startlux-decide.)
4. planner: groups with images -> positions, the mrope table, the image tokens' ids (-1) and feature runs; the
   prefix cache and the sharing between requests keyed by the images' digests.
5. requests through Decider (a fake backend): every row of a request holds the images, wide choices included;
   usage counts the image tokens; refusals (text-only servers, too many images, marks that do not match) with v1.1's
   messages and locations.
"""
import argparse
import base64
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from startlux_prism import decide as D  # noqa: E402
from startlux_prism import images as I  # noqa: E402
from startlux_prism import jevfmt as J  # noqa: E402
from startlux_prism.engine import batch as B  # noqa: E402
from startlux_prism.engine.prefix_cache import PrefixCache  # noqa: E402

QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Does this delivery need a follow-up today?"},
    "shape": {"type": "choice", "instructions": "Which shape is the largest?",
              "criteria": {"circle": "a round shape", "square": None, "triangle": "three sides"}},
    "count": {"type": "score", "instructions": "How many shapes are there?", "criteria": ["one", "two", "three"]}}


def png(width, height, color=(200, 30, 30), uri=True, kind="PNG", draw=False):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (width, height), color)
    if draw:
        d = ImageDraw.Draw(img)
        d.ellipse((width // 8, height // 8, width // 2, height // 2), fill=(20, 200, 40))
        d.rectangle((width // 2, height // 3, width - 5, height - 5), fill=(30, 30, 220))
        d.text((5, 5), "INVOICE 2291", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, kind)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{kind.lower()};base64,{b64}" if uri else b64


def raises(fn, text, loc=None):
    try:
        fn()
    except I.ImageError as e:
        assert text in str(e), (text, str(e))
        assert loc is None or e.loc == loc, (loc, e.loc)
        return
    raise AssertionError(f"no error ({text})")


def check_requests():
    S = I.SLOT
    assert I.place_images(f"before {I.IMAGE_MARK} after", 1) == f"before {S} after"
    assert I.place_images(f"{I.IMAGE_MARK}{I.IMAGE_MARK}", 2) == S + S
    assert I.place_images("the parcel", 1) == f"{S}\n\nthe parcel"
    assert I.place_images("the parcel", 2) == f"Image 1: {S}\nImage 2: {S}\n\nthe parcel"
    assert I.place_images("", 1) == S
    assert I.place_images('{"photo": "<image>", "id": 7}', 1) == '{"photo": "' + S + '", "id": 7}'
    raises(lambda: I.place_images(f"{I.IMAGE_MARK} and {I.IMAGE_MARK}", 1), "2 <image> marks for 1 images",
           ["body", "state"])
    assert I.smart_resize(1024, 1024) == (1024, 1024)
    assert I.smart_resize(10, 10) == (256, 256)
    h, w = I.smart_resize(3000, 2000)
    assert h % 32 == 0 and w % 32 == 0 and h * w <= 1 << 20 and abs(h / w - 1.5) < 0.05
    assert I.smart_resize(4000, 4000, max_pixels=1 << 22) == (2048, 2048)
    raises(lambda: I.smart_resize(10, 3000), "factor of 200")
    m = I.load_images([png(1024, 1024), png(64, 32, uri=False), png(300, 200, kind="JPEG")])
    assert [x.rgb.shape[:2] for x in m] == [(1024, 1024), (192, 384), (224, 320)], [x.rgb.shape for x in m]
    assert [x.tokens for x in m] == [1024, 72, 70] and (m[1].rows, m[1].cols) == (6, 12)
    assert len({x.digest for x in m}) == 3 and m[0].digest == I.load_images([png(1024, 1024)])[0].digest
    for bad, why in (("not base64 !", "base64"), (base64.b64encode(b"text").decode(), "not a readable image"),
                     (7, "base64 string")):
        raises(lambda: I.load_images([bad]), why, ["body", "images"])
    raises(lambda: I.load_images([png(40, 40)] * 3, I.Settings(max_images=2)), "at most 2 images")
    raises(lambda: I.load_images(png(40, 40)), "a list")
    # base64: the fast strict path accepts exactly what v1.1's rule (whitespace removed, then validated) accepts
    import binascii
    import re
    b64 = png(30, 20, uri=False)

    def v11(value):
        m = I._DATA_URI.match(value)
        try:
            return base64.b64decode(re.sub(r"\s+", "", value[m.end():] if m else value), validate=True)
        except (binascii.Error, ValueError):
            return "refused"

    def ours(value):
        try:
            return I.decode(value)
        except I.ImageError:
            return "refused"
    cases = [b64, "data:image/png;base64," + b64, "data:;base64," + b64, b64[:40] + "\n" + b64[40:] + "\n",
             " " + b64, b64.rstrip("=") or b64, b64[:-1], b64 + "=", b64 + "==", "", "QQ==", "QQ=", "QUI=", "QUJD",
             "QQ==QQ==", "Q-_=", "Qé==", "data:image/png;base64,", "!!!!", "QQ==\n"]
    for c in cases:
        assert ours(c) == v11(c), (c[:50], ours(c)[:20], v11(c)[:20])
    print("1. requests: places, resize rule, decoding and errors as StartLux-Decision v1.1", flush=True)


def check_pixels(model_dir):
    import torch
    from startlux_prism.vision import Config, VisionTower
    try:
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
    except ImportError:
        print("2. pixels: skipped (no transformers)")
        return
    import json
    cfg = json.load(open(os.path.join(model_dir, "preprocessor_config.json")))
    proc = Qwen2VLImageProcessorPil(**{k: v for k, v in cfg.items() if k not in ("processor_class",
                                                                                "image_processor_type")})
    tower = VisionTower.__new__(VisionTower)            # the input half only, on the CPU
    tower.cfg, tower.device = Config(blocks=0, hidden=1152, heads=16, ffn=0, out=0), torch.device("cpu")
    tower.mean = torch.tensor(tower.cfg.mean)
    tower.std = torch.tensor(tower.cfg.std)
    n = 0
    for w, h, kind in ((640, 480, "PNG"), (333, 777, "JPEG"), (1500, 900, "PNG"), (50, 40, "PNG"), (2048, 64, "JPEG")):
        value = png(w, h, color=(w % 256, h % 256, 77), kind=kind, draw=True)
        image = I.load_images([value])[0]
        from PIL import Image as PILImage
        raw = PILImage.open(io.BytesIO(base64.b64decode(value.split(",", 1)[1]))).convert("RGB")
        ref = proc(images=[raw], size={"longest_edge": I.MAX_PIXELS, "shortest_edge": I.MIN_PIXELS},
                   return_tensors="pt")
        gh, gw = image.rgb.shape[0] // 16, image.rgb.shape[1] // 16
        assert ref["image_grid_thw"].tolist() == [[1, gh, gw]], (ref["image_grid_thw"], gh, gw)
        mine = tower.patches(image)
        assert torch.equal(mine, ref["pixel_values"].to(torch.bfloat16)), (w, h)
        n += 1
    print(f"2. pixels: {n} images, the tower's input bit-identical to transformers' image processor", flush=True)


def walk_positions(ids, pad, images):
    """Positions by walking the tokens one at a time (independent of images.positions)."""
    out, cur, k, i = [], 0, 0, 0
    while i < len(ids):
        if ids[i] != pad:
            out.append((cur, cur, cur))
            cur += 1
            i += 1
            continue
        im = images[k]
        k += 1
        for r in range(im.rows):
            for c in range(im.cols):
                out.append((cur, cur + r, cur + c))
        cur += max(im.rows, im.cols)
        i += im.tokens
    return np.asarray(out, np.int64).T


def check_prompts(model_dir, startlux):
    from startlux_prism.prompt import Tokenizer
    tok = Tokenizer(model_dir)
    settings = I.Settings()
    pad = settings.pad_id
    requests = [("Photo taken at delivery: <image>", [png(320, 256)]),
                ("the parcel", [png(640, 480, draw=True)]),
                ("first <image> then <image>", [png(64, 64), png(96, 64, color=(0, 90, 0))]),
                ({"ticket": "damaged box", "photo": "<image>", "items": list(range(9))}, [png(1000, 300, draw=True)]),
                ("", [png(200, 200), png(500, 260, kind="JPEG"), png(48, 900)]),
                ("eight images", [png(64 + 32 * k, 64, color=(k * 30, 0, 0)) for k in range(8)])]
    ref_tok = get_rope_index = None
    try:
        import torch
        from transformers import AutoTokenizer
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
        ref_tok = AutoTokenizer.from_pretrained(model_dir)

        class Cfg:
            class vision_config:
                spatial_merge_size = 2

        class Fake:
            config = Cfg
            get_vision_position_ids = staticmethod(lambda *a, **k: Qwen3_5Model.get_vision_position_ids(None, *a, **k))

        def get_rope_index(ids, images):
            ids_t = torch.tensor([ids])
            grid = torch.tensor([[1, im.rows * 2, im.cols * 2] for im in images])
            pos, delta = Qwen3_5Model.get_rope_index(Fake, ids_t, (ids_t == pad).int(), image_grid_thw=grid)
            return pos[:, 0].numpy(), int(delta)
    except ImportError:
        print("   (no transformers: positions checked against the token walk only)")
    P = None
    if startlux:
        sys.path.insert(0, startlux)
        from startlux_decide import prompt as P
    n_rows = n_ref = 0
    for state, values in requests:
        imgs = I.load_images(values)
        text = I.place_images(J.state_text(state), len(imgs))
        specs = [(q, none) for q in QUESTIONS.values() for none in (False, True)]
        recs = [J.from_systemone(text, q, allow_none=none) for q, none in specs]
        rows = I.rows_for(tok, recs, [f"q{i}" for i in range(len(recs))], imgs, settings)
        for r, (q, none) in zip(rows, specs):
            n_rows += 1
            ids = np.asarray(r.ids)
            assert (ids == pad).sum() == sum(im.tokens for im in imgs)
            assert len({x.head for x in rows}) == 1 and r.ids[:r.head] == rows[0].ids[:r.head]
            for off, im in r.images:
                assert (ids[off:off + im.tokens] == pad).all() and ids[off - 1] == 248053 and \
                    ids[off + im.tokens] == 248054
            want = walk_positions(r.ids, pad, imgs)
            assert (r.pos == want).all()
            assert r.last_pos == r.pos.max() and (r.pos[:, -1] == r.last_pos).all()
            if get_rope_index is not None:
                ref_pos, delta = get_rope_index(r.ids, imgs)
                assert (ref_pos == r.pos).all() and r.last_pos + 1 == len(r.ids) + delta
            if P is not None and ref_tok is not None:
                prompt = P.plain_prompt(ref_tok, P.messages(P.from_systemone(text, q, none)))
                parts = prompt.split(I.IMAGE_PAD)
                assert len(parts) == len(imgs) + 1
                expanded = parts[0] + "".join(I.IMAGE_PAD * im.tokens + p for im, p in zip(imgs, parts[1:]))
                assert ref_tok(expanded, add_special_tokens=False)["input_ids"] == r.ids, state
                n_ref += 1
    print(f"3. prompts: {n_rows} rows with 1-8 images; positions as get_rope_index{'' if get_rope_index else ' (n/a)'} "
          f"and the token walk; {n_ref} prompts token-identical to startlux_decide's", flush=True)


SYS = list(range(100, 130))


def fake_rows(rng, images, n_q=2, ev_len=40):
    """Rows with images in the evidence: SYS, text, image tokens..., text, then a question tail each."""
    pad = 248056
    ev, offsets = [], []
    for im in images:
        ev += list(rng.integers(1000, 2000, 5)) + [248053]
        offsets.append(len(SYS) + len(ev))
        ev += [pad] * im.tokens + [248054]
    ev += list(rng.integers(1000, 2000, ev_len))
    head = len(SYS) + len(ev)
    rows, pos = [], []
    hp, nxt = I.positions(np.asarray(SYS + ev), offsets, images)
    for q in range(n_q):
        tail = list(rng.integers(3000, 4000, 5 + q))
        rows.append(SYS + ev + tail)
        pos.append(np.concatenate([hp, np.broadcast_to(nxt + np.arange(len(tail)), (3, len(tail)))], 1))
    return B.Group(rows, head, pos, tuple(zip(offsets, images)))


def check_planner():
    rng = np.random.default_rng(0)
    img = lambda r, c, seed: I.Image(np.zeros((32 * r, 32 * c, 3), np.uint8), bytes([seed]) * 24)  # noqa: E731
    a, b = img(3, 5, 1), img(2, 2, 2)
    g = fake_rows(rng, [a, b])
    plan = B.plan_batch([g], len(SYS), 64, (0, 30))
    t = plan.tables
    # every image token but each image's first (row 0, column 0: the same position on all three axes) is in the table
    assert plan.counts["segs"] == 3 and plan.counts["mrope"] == a.tokens - 1 + b.tokens - 1
    e = plan.segs[0]                                   # the shared evidence: every image token in it
    want = g.pos[0][:, len(SYS):g.head]
    for j in range(len(e.tokens)):
        p = t["pos"][e.start + j]
        if p >= 0:
            assert want[0, j] == want[1, j] == want[2, j] == p
        else:
            assert tuple(t["mrope"][-1 - p]) == tuple(want[:, j])
    image_tok = np.flatnonzero(t["ids"][:plan.n_tok] == -1)
    assert len(image_tok) == a.tokens + b.tokens
    runs = sorted((s, im.tokens, j0, n) for s, im, j0, n in plan.feats)
    assert [(n, j0) for _, _, j0, n in runs] == [(a.tokens, 0), (b.tokens, 0)]
    assert sorted(image_tok.tolist()) == sorted(s + k for s, _, _, n in runs for k in range(n))
    for k, si in enumerate(plan.read_seg):              # question tails continue after the evidence's positions
        s = plan.segs[si]
        assert s.pos0 >= g.head and (t["pos"][s.start:s.start + len(s.tokens)] == g.pos[k][0, s.pos0:]).all()
        assert t["read_pos"][k] == g.pos[k][0, -1]
    # prefix cache: the same tokens with another image are another entry; the same image is a hit
    S = np.asarray(SYS, np.int32)
    cache = PrefixCache(slots=range(2, 6), kv_start=10000, kv_rows=10000, min_tokens=8)
    g2 = B.Group(g.rows, g.head, g.pos, ((g.images[0][0], img(3, 5, 9)), g.images[1]))
    p1 = B.plan_batch([g, g2], len(SYS), 64, (0, 30), cache=cache, system=S)
    assert len(p1.inserted) == 2 and cache.stats["batch_shared"] == 0 and len(p1.feats) == 4
    p2 = B.plan_batch([g], len(SYS), 64, (0, 30), cache=cache, system=S)
    assert cache.stats["hits"] == 1 and not p2.feats and p2.counts["mrope"] == 0 and (p2.tables["ids"] >= 0).all()
    assert (p2.tables["pos"] == np.concatenate([g.pos[k][0, g.head:] for k in range(2)])).all()
    p3 = B.plan_batch([fake_rows(np.random.default_rng(5), [a]), fake_rows(np.random.default_rng(5), [a])],
                      len(SYS), 64, (0, 30), cache=cache, system=S)
    assert cache.stats["batch_shared"] == 1 and len(p3.feats) == 1
    # without a cache the rows of a request share the evidence through its common prefix, images and all
    p4 = B.plan_batch([fake_rows(rng, [a, b, a], n_q=3)], len(SYS), 64, (0, 30))
    assert p4.counts["segs"] == 4 and len(p4.feats) == 3
    print("4. planner: positions, mrope table, image tokens and their features; cache and sharing keyed by images",
          flush=True)


class FakeBackend:
    def __init__(self):
        self.rows = []

    def letter_logits(self, rows):
        self.rows.append(rows)
        return [[float((len(r.ids) * 7 + i * 3) % 11) for i in range(r.n_options)] for r in rows]


def check_decider(model_dir):
    from startlux_prism.prompt import Tokenizer
    tok = Tokenizer(model_dir)
    settings = I.Settings(max_images=2)
    be = FakeBackend()
    d = D.Decider(tok, be, images=settings)
    imgs = [png(320, 256), png(64, 64, color=(0, 0, 255))]
    wide = {"type": "choice", "instructions": "Which part?", "criteria": {f"p{i}": None for i in range(40)}}
    answers, usage = d.decide("Photo: <image>, label: <image>", dict(QUESTIONS, wide=wide), images=imgs)
    assert set(answers) == set(QUESTIONS) | {"wide"}
    n_img = sum(im.tokens for im in I.load_images(imgs))
    assert n_img == 80 + 64                             # 320 x 256; 64 x 64 resized up to 256 x 256
    for rows in be.rows:
        assert all(len(r.images) == 2 and (np.asarray(r.ids) == settings.pad_id).sum() == n_img for r in rows)
    assert usage["input_tokens"] == sum(len(r.ids) for rows in be.rows for r in rows)
    assert len(be.rows) == 3                            # the wide question's two rounds, then the others
    text = D.Decider(tok, FakeBackend()).decide("Photo: x", QUESTIONS)[0]
    assert text.keys() == answers.keys() - {"wide"}
    D.Decider(tok, FakeBackend(), images=settings).decide("no images", QUESTIONS, images=[])
    for dec, images, state, why, loc in (
            (D.Decider(tok, FakeBackend()), imgs[:1], "x", "reads text only", ["body", "images"]),
            (D.Decider(tok, FakeBackend(), images=I.Settings(refused="this model folder has no vision tower: it "
                                                                     "reads text only")),
             imgs[:1], "x", "no vision tower", ["body", "images"]),
            (d, imgs * 2, "x", "at most 2 images", ["body", "images"]),
            (d, imgs[:1], "<image> <image>", "2 <image> marks for 1 images", ["body", "state"]),
            (d, ["%%%"], "x", "base64", ["body", "images"]),
            (d, "x", "x", "a list", ["body", "images"]),
            (d, imgs[:1], "x <|image_pad|>", "<|image_pad|> tokens for 1 images", ["body", "images"])):
        raises(lambda: dec.decide(state, QUESTIONS, images=images), why, loc)
    print("5. requests: every row and round holds the images, image tokens counted, refusals as v1.1", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="a StartLux-Decision v1.1 folder with a vision tower")
    ap.add_argument("--startlux-decide", default=os.environ.get("STARTLUX_DECIDE", ""),
                    help="StartLux-Decision v1.1's package (Startlux-decide-inference) for the prompt check")
    a = ap.parse_args()
    check_requests()
    check_pixels(a.model_dir)
    check_prompts(a.model_dir, a.startlux_decide)
    check_planner()
    check_decider(a.model_dir)
    print("all ok")


if __name__ == "__main__":
    main()
