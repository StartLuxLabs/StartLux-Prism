"""StartLux-Prism with images against transformers' Qwen3.5 forward on the same weights: the reference engine of
StartLux-Decision v1.1's transformers backend (startlux_decide/backends/transformers.py), which reads a prompt with
images through the model's image processor, its vision tower and get_rope_index's positions.

    # the vision tower alone: StartLux-Prism's (vision.py) against transformers' Qwen3_5VisionModel, the mmproj's weights
    python tests/vision_reference.py encoder --model-dir V11_DIR --mmproj MMPROJ.gguf
    # the reference: a bf16 checkpoint of the GGUF and the mmproj (tools/gguf_to_hf.py --mmproj), one row per forward
    python tests/vision_reference.py transformers --hf HF_DIR --model-dir V11_DIR --out ref.pt
    # StartLux-Prism (the vision tower: --mmproj, default the mmproj-*.gguf beside the GGUF)
    python tests/vision_reference.py prism --gguf X.gguf --model-dir V11_DIR --precision int8 --calib C --out f.pt
    python tests/vision_reference.py compare ref.pt f.pt [g.pt ...]

encoder: images of several sizes and aspect ratios, each through both towers (transformers' input from its own image
processor, the Pillow backend): the features must be the same bits.
transformers / StartLux-Prism: both sides answer the same requests through StartLux-Prism's Decider (decide.py, checked against
startlux_decide's read-out by tests/test_v11.py): bench/vision_suite.py's image requests (1 to 8 images, marks and none,
32 x 32 to 2,400 x 1,600 pixels) and, as the yardstick for how close two implementations of the model get, the text-only
requests of bench/suites.py.  A run records every answer and the letter logits of every prompt row.  compare reports,
per suite, the largest and mean difference of the answers' probabilities, how often the decisions agree, and each run's
score against the suite's right answers.
"""
import argparse
import importlib.util
import io
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from startlux_prism import images as I  # noqa: E402
from startlux_prism.decide import Decider  # noqa: E402
from startlux_prism.prompt import Tokenizer  # noqa: E402


def _bench(name):
    """A module of bench/ by its path (bench/ on sys.path would shadow the standard library's profile module, which
    transformers imports through cProfile)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), "..", "bench",
                                                                     name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


suites, vision_suite = _bench("suites"), _bench("vision_suite")


class Recorder:
    """Wraps a backend: keeps every row's letter logits (key -> list of fp32 tensors, in call order)."""

    def __init__(self, backend, name):
        self.backend, self.name, self.logits = backend, name, {}

    def letter_logits(self, rows):
        out = self.backend.letter_logits(rows)
        for r, z in zip(rows, out):
            self.logits.setdefault(f"{self.name}/{r.key}", []).append(torch.as_tensor(z)[:r.n_options].float().cpu())
        return out


class Transformers:
    """transformers' Qwen3_5ForConditionalGeneration, bf16, one prompt row per forward pass, read as the package's
    transformers backend reads it: the last position's hidden state (after the final norm) times the output head's
    letter rows, in fp32.  Images: the image processor (its Pillow backend; transformers' default without torchvision)
    on the request's decoded images, within the request's size bounds."""

    def __init__(self, hf_dir, model_dir, letters, device="cuda"):
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
        self.device = torch.device(device)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            hf_dir, dtype=torch.bfloat16, device_map={"": self.device}, attn_implementation="sdpa").eval()
        self.head = self.model.get_output_embeddings().weight
        self.letters = torch.tensor(letters, device=self.device)
        cfg = json.load(open(os.path.join(model_dir, "preprocessor_config.json")))
        self.proc = Qwen2VLImageProcessorPil(**{k: v for k, v in cfg.items()
                                                 if k not in ("processor_class", "image_processor_type")})
        self.pad = self.model.config.image_token_id
        self.originals = {}                     # image digest -> the decoded PIL image as sent

    def add_images(self, values):
        from PIL import Image
        for v, im in zip(values, I.load_images(values)):
            self.originals[im.digest] = Image.open(io.BytesIO(I.decode(v))).convert("RGB")

    @torch.no_grad()
    def letter_logits(self, rows):
        out = []
        for r in rows:
            ids = torch.tensor([r.ids], device=self.device)
            if r.images:
                enc = self.proc(images=[self.originals[im.digest] for _, im in r.images],
                                size={"longest_edge": I.MAX_PIXELS, "shortest_edge": I.MIN_PIXELS}, return_tensors="pt")
                hidden = self.model.model(input_ids=ids, pixel_values=enc["pixel_values"].to(self.device),
                                          image_grid_thw=enc["image_grid_thw"].to(self.device),
                                          mm_token_type_ids=(ids == self.pad).int(), use_cache=False,
                                          return_dict=True).last_hidden_state
            else:
                hidden = self.model.model.language_model(input_ids=ids, use_cache=False,
                                                         return_dict=True).last_hidden_state
            rows_ = self.head.index_select(0, self.letters)
            out.append((hidden[0, -1].float() @ rows_.float().T).cpu())
        return out


def run(a):
    tok = Tokenizer(a.model_dir)
    if a.engine == "transformers":
        backend = Transformers(a.hf, a.model_dir, tok.letters)
    else:
        from startlux_prism.engine.model import Engine, Options
        from startlux_prism.vision import find_mmproj
        mmproj, why = find_mmproj(a.mmproj, a.gguf, a.model_dir)
        if not mmproj:
            sys.exit(why)
        backend = Engine(a.gguf, a.model_dir, Options(precision=a.precision, calib=a.calib, max_tokens=a.max_tokens,
                                                      mmproj=mmproj, prefix_cache=a.prefix_cache))
        if a.warm:
            backend.warm_up()
    settings = I.Settings()
    reqs = [("image", name, state, qs, imgs, right) for name, state, qs, imgs, right in vision_suite.requests()]
    reqs += [("text", name, state, qs, [], None) for name, (state, qs) in suites.fixed().items() if name != "wide33"]
    res = {"answers": {}, "logits": {}, "right": {}, "suite": {}, "ms": {}}
    for suite, name, state, qs, imgs, right in reqs:
        if imgs and hasattr(backend, "add_images"):
            backend.add_images(imgs)
        rec = Recorder(backend, name)
        t0 = time.perf_counter()
        answers, usage = Decider(tok, rec, images=settings).decide(state, qs, images=imgs or None)
        res["ms"][name] = (time.perf_counter() - t0) * 1000
        res["answers"][name], res["right"][name], res["suite"][name] = answers, right, suite
        res["logits"].update(rec.logits)
        print(f"{name}: {usage['input_tokens']} tokens, {res['ms'][name]:.0f} ms", json.dumps(answers)[:160],
              flush=True)
    torch.save(res, a.out)
    print(f"{len(reqs)} requests -> {a.out}")


def probs(ans):
    """An answer -> its probabilities as a flat list (with none_probability) and its decision."""
    if ans["type"] == "noul":
        return [ans["noul"]], "true" if ans["noul"] >= 0.5 else "false"
    p = list(ans["probabilities"].values()) + ([ans["none_probability"]] if "none_probability" in ans else [])
    best = max(range(len(ans["probabilities"])), key=lambda i: list(ans["probabilities"].values())[i])
    key = list(ans["probabilities"])[best]
    return p, (int(key) if ans["type"] == "score" else key)


def compare(paths):
    runs = [torch.load(p) for p in paths]
    ref = runs[0]
    print(f"reference: {paths[0]}")
    for path, r in zip(paths, runs):
        for suite in ("image", "text"):
            names = [n for n in ref["answers"] if ref["suite"][n] == suite]
            if r is not ref:
                dmax, dsum, n_p, same, n_q = 0.0, 0.0, 0, 0, 0
                worst = None
                for n in names:
                    for k, ans in ref["answers"][n].items():
                        pa, da = probs(ans)
                        pb, db = probs(r["answers"][n][k])
                        d = [abs(x - y) for x, y in zip(pa, pb)]
                        if max(d) > dmax:
                            dmax, worst = max(d), f"{n}/{k}"
                        dsum += sum(d)
                        n_p += len(d)
                        same += da == db
                        n_q += 1
                zmax = max((float((za - zb).abs().max()) for key, zs in ref["logits"].items()
                            if key.split("/")[0] in names for za, zb in zip(zs, r["logits"][key])), default=0.0)
                print(f"  {path} [{suite}]: {n_q} questions, decisions the same on {same}; probabilities: largest "
                      f"difference {dmax:.4f} ({worst}), mean {dsum / max(n_p, 1):.5f}; letter logits: largest "
                      f"difference {zmax:.3f}")
            if suite == "image":
                score = n_q = 0
                for n in names:
                    for k, ans in r["answers"][n].items():
                        score += probs(ans)[1] == ref["right"][n][k]
                        n_q += 1
                ms = sum(r["ms"][n] for n in names)
                print(f"  {path} [image]: {score} of {n_q} right; {ms:.0f} ms for the {len(names)} requests")


def encoder(a):
    """The two vision towers on the same images (module docstring)."""
    from PIL import Image
    from transformers import AutoConfig
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    from startlux_prism.vision import VisionTower, find_mmproj, load
    mmproj = find_mmproj(a.mmproj, a.gguf or os.path.join(a.model_dir, "x.gguf"), a.model_dir)[0]
    dev = torch.device("cuda")
    tower = VisionTower(mmproj, dev)
    vc = AutoConfig.from_pretrained(a.model_dir).vision_config
    vc._attn_implementation = "sdpa"
    # built on the CPU, then moved: its rotary frequencies computed on the CPU, as from_pretrained has them
    ref = Qwen3_5VisionModel._from_config(vc, dtype=torch.bfloat16)
    ref.load_state_dict({k.removeprefix("model.visual."): t.to(torch.bfloat16) for k, t in load(mmproj)[1].items()})
    ref = ref.to(dev).eval()
    cfg = json.load(open(os.path.join(a.model_dir, "preprocessor_config.json")))
    proc = Qwen2VLImageProcessorPil(**{k: v for k, v in cfg.items()
                                       if k not in ("processor_class", "image_processor_type")})
    sizes =((256, 256), (640, 480), (333, 777), (1024, 1024), (1500, 900), (2048, 64), (64, 2048), (40, 30),
             (2400, 1600))
    from torch.nn.attention import sdpa_kernel
    from startlux_prism.vision import ATTENTION
    bad = 0
    for w, h in sizes:
        value = vision_suite.shapes("red", "blue", w, h)
        image = I.load_images([value])[0]
        enc = proc(images=[Image.open(io.BytesIO(I.decode(value))).convert("RGB")],
                   size={"longest_edge": I.MAX_PIXELS, "shortest_edge": I.MIN_PIXELS}, return_tensors="pt")
        pv, grid = enc["pixel_values"].to(dev).to(torch.bfloat16), enc["image_grid_thw"].to(dev)
        with torch.no_grad():
            r_default = ref(pv, grid_thw=grid).pooler_output                    # transformers as it runs (cuDNN's)
            with sdpa_kernel(ATTENTION):
                r = ref(pv, grid_thw=grid).pooler_output                        # with StartLux-Prism's attention kernel
            mine = [tower.encode(image) for _ in range(3)]          # eager, then (small sizes) recorded and replayed
        same = all(torch.equal(m, r) for m in mine)
        bad += not same
        d = (mine[0].float() - r_default.float()).abs()
        print(f"{w} x {h} -> {image.rgb.shape[1]} x {image.rgb.shape[0]}, {image.tokens} tokens: "
              f"{'bit-identical' if same else 'DIFFERENT'} with the same attention kernel; against transformers' "
              f"default kernel max abs difference {d.max().item():.3g} (relative to the largest feature "
              f"{d.max().item() / r_default.float().abs().max().item():.2g}), mean {d.mean().item():.2g}", flush=True)
    print("all bit-identical" if not bad else f"{bad} DIFFER")
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=("encoder", "transformers", "prism", "compare"))
    ap.add_argument("runs", nargs="*", help="compare: the reference run first")
    ap.add_argument("--hf", help="transformers: the bf16 checkpoint (tools/gguf_to_hf.py --mmproj)")
    ap.add_argument("--gguf")
    ap.add_argument("--model-dir")
    ap.add_argument("--mmproj", default="auto")
    ap.add_argument("--precision", default="int8")
    ap.add_argument("--calib", default="")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--prefix-cache", type=int, default=64)
    ap.add_argument("--warm", action="store_true", help="StartLux-Prism: record every graph first (steady-state times)")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.engine == "compare":
        compare(a.runs)
    elif a.engine == "encoder":
        encoder(a)
    else:
        run(a)


if __name__ == "__main__":
    main()
