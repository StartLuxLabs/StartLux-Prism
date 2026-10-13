"""Which kind of StartLux-Decision model folder this is, by the rules of StartLux-Decision v1.1's package
(startlux_decide/modelcfg.py, Apache-2.0):

    v1          decision_config.json "format": "StartLux-Decision-v1" (and no "auto_think" block): one pass
    auto_think  v1.1 AutoThink (causal): reads the answer once and, in the reference, thinks when unsure
    instant     v1.1 Instant: config.json "jev_prompt_attention": "bidirectional" (a GGUF of it carries
                qwen35.attention.causal = false): the prompt is read bidirectionally in the full-attention layers

flash-jev answers in one pass (think_mode "never") and runs the prompt causally, so it serves v1 and AutoThink folders
and refuses Instant ones.  What the kind changes in the request handling (decide.py): v1 folders answer exactly as
StartLux-Decision v1 does; v1.1 folders follow v1.1 (none_check on by default, the knock-out rule for long choice lists,
the read-out's order of operations).
"""
from __future__ import annotations

import json
import os

V1_FORMAT = "StartLux-Decision-v1"
KINDS = ("v1", "auto_think", "instant")
PROMPT_ATTENTION_KEY = "jev_prompt_attention"


def folder_kind(model_dir, gguf_meta=None):
    """"v1", "auto_think" or "instant"; raises ValueError for an inconsistent folder.  gguf_meta: the GGUF file's
    metadata (a non-causal file is Instant whatever the folder says)."""
    with open(os.path.join(model_dir, "decision_config.json")) as f:
        cfg = json.load(f)
    mode = "causal"
    path = os.path.join(model_dir, "config.json")
    if os.path.isfile(path):
        with open(path) as f:
            mode = json.load(f).get(PROMPT_ATTENTION_KEY, "causal")
    if mode not in ("causal", "bidirectional"):
        raise ValueError(f"{path}: {PROMPT_ATTENTION_KEY} must be causal or bidirectional, not {mode!r}")
    declared = cfg.get("prompt_attention", mode)
    if declared != mode:
        raise ValueError(f"{model_dir}: prompt attention is {mode!r} in config.json but {declared!r} in "
                         "decision_config.json")
    if gguf_meta is not None and gguf_meta.get("qwen35.attention.causal") is False:
        mode = "bidirectional"
    if mode == "bidirectional":
        return "instant"
    return "v1" if cfg.get("format") == V1_FORMAT and "auto_think" not in cfg else "auto_think"
