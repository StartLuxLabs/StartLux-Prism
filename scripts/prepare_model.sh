#!/bin/bash
# Every model folder the comparison needs, from one StartLux-Decision GGUF repository (run once per model on the GPU
# server; the 4B's folders were made the same way):
#   HF           bf16 safetensors, Q8_0 values dequantized exactly (flash_jev/tools/gguf_to_hf.py): StartLux-Decision's
#                own server, and the weights of every folder below
#   CAUSAL       the same weights as Qwen3_5ForCausalLM (the text config; symlinks): vLLM, the adapter's tokenizer
#   CAUSAL_FLAT  a copy with keys model.* instead of model.language_model.*: SGLang (its loader leaves the
#                model.language_model.* weights uninitialized, silently)
#   DECIDER_SHAPE, JEVK5_SHAPE   decider's and JevK5's own configs with this model's sizes over its weights (symlinks):
#                the same-shape speed stand-ins (they render their own prompts; their answers are not StartLux's)
#
#   MODEL=~/jev/models/StartLux-Decision-9B-Q8_0-GGUF bash scripts/prepare_model.sh
#
# Prints the environment for scripts/compare.sh, baselines.sh and stress.sh.  Folders already there are kept.
set -euo pipefail
JEV=${JEV:-$HOME/jev}
cd "$(dirname "$0")/.."
source "$JEV/env.sh"
MODEL=${MODEL:?MODEL: the GGUF repository directory}
GGUF=${GGUF:-$(ls "$MODEL"/*.gguf | head -1)}
NAME=$(basename "$MODEL" | sed 's/-Q8_0-GGUF$//; s/-GGUF$//')            # StartLux-Decision-9B
M=$JEV/models
HF=${HF:-$M/$NAME-bf16-from-Q8_0}
CAUSAL=${CAUSAL:-$M/$NAME-causallm}
CAUSAL_FLAT=${CAUSAL_FLAT:-$M/$NAME-causallm-flat}
DECIDER_SHAPE=${DECIDER_SHAPE:-$M/decider-shape-$NAME}
JEVK5_SHAPE=${JEVK5_SHAPE:-$M/jevk5-shape-$NAME}
REF=$M/StartLux-Decision-4B-bf16-from-Q8_0          # the 4B checkpoint: the stand-ins' configs are the 4B's plus the
                                                    # keys where this model's text config differs from the 4B's

if [ ! -f "$HF/model.safetensors.index.json" ]; then
  echo "converting $GGUF -> $HF"
  python -m flash_jev.tools.gguf_to_hf --gguf "$GGUF" --model-dir "$MODEL" --out "$HF"
fi

if [ ! -f "$CAUSAL/config.json" ]; then
  mkdir -p "$CAUSAL"
  for f in "$HF"/model-*.safetensors "$HF"/model.safetensors.index.json "$HF"/{tokenizer.json,tokenizer_config.json,vocab.json,merges.txt,chat_template.jinja,decision_config.json}; do
    if [ -e "$f" ]; then ln -sf "$f" "$CAUSAL/"; fi
  done
  python - "$HF/config.json" "$CAUSAL/config.json" <<"PY"
import json, sys
c = json.load(open(sys.argv[1]))
t = dict(c["text_config"])
t["architectures"], t["model_type"] = ["Qwen3_5ForCausalLM"], "qwen3_5_text"
t["tie_word_embeddings"] = bool(c.get("tie_word_embeddings", t.get("tie_word_embeddings", True)))
json.dump(t, open(sys.argv[2], "w"), indent=1)
PY
fi

if [ ! -f "$CAUSAL_FLAT/model.safetensors.index.json" ]; then
  echo "writing $CAUSAL_FLAT (keys model.*)"
  python - "$CAUSAL" "$CAUSAL_FLAT" <<"PY"
import json, os, shutil, sys
from safetensors import safe_open
from safetensors.torch import save_file
src, dst = sys.argv[1:3]
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if not f.endswith(".safetensors") and f != "model.safetensors.index.json":
        shutil.copy(os.path.join(src, f), dst)
idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
ren = lambda k: k.replace("model.language_model.", "model.", 1)              # noqa: E731
for shard in sorted(set(idx["weight_map"].values())):
    with safe_open(os.path.join(src, shard), "pt") as f:
        t = {ren(k): f.get_tensor(k) for k in f.keys()}
        meta = f.metadata()
    save_file(t, os.path.join(dst, shard), metadata=meta or {"format": "pt"})
idx["weight_map"] = {ren(k): v for k, v in idx["weight_map"].items()}
json.dump(idx, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=1)
PY
fi

standin() {   # template folder (the 4B stand-in) -> new folder with this model's sizes over its weights
  local tpl=$1 out=$2
  if [ -f "$out/config.json" ]; then return; fi
  mkdir -p "$out"
  for f in "$tpl"/*; do
    case "$(basename "$f")" in
      model*.safetensors|model.safetensors.index.json) ;;
      config.json) ;;
      *) cp -L "$f" "$out/" ;;
    esac
  done
  for f in "$HF"/model-*.safetensors "$HF"/model.safetensors.index.json; do ln -sf "$f" "$out/"; done
  python - "$tpl/config.json" "$REF/config.json" "$HF/config.json" "$out/config.json" <<"PY"
import json, sys
tpl, ref, new, out = sys.argv[1:5]
c = json.load(open(tpl))
r, n = json.load(open(ref)), json.load(open(new))
rt, nt = r["text_config"], n["text_config"]
changed = {k: nt[k] for k in nt if rt.get(k) != nt[k]}
changed["tie_word_embeddings"] = bool(n.get("tie_word_embeddings", nt.get("tie_word_embeddings", True)))
c.update(changed)
json.dump(c, open(out, "w"), indent=1)
print(f"{out}: {sorted(changed)}")
PY
}
standin "$M/decider-shape-startlux" "$DECIDER_SHAPE"
standin "$M/jevk5-shape-startlux" "$JEVK5_SHAPE"

cat <<EOF
# environment for scripts/compare.sh, baselines.sh, stress.sh:
export MODEL=$MODEL HF=$HF CAUSAL=$CAUSAL CAUSAL_FLAT=$CAUSAL_FLAT DECIDER_SHAPE=$DECIDER_SHAPE JEVK5_SHAPE=$JEVK5_SHAPE
EOF
