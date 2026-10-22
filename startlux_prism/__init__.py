"""StartLux-Prism: a CUDA inference engine for StartLux-Decision models (formerly flash-jev)."""
import os as _os

# the engine's former name: a FLASHJEV_* environment variable still applies where its STARTLUX_PRISM_* one is unset
for _k, _v in list(_os.environ.items()):
    if _k.startswith("FLASHJEV_"):
        _os.environ.setdefault("STARTLUX_PRISM_" + _k[len("FLASHJEV_"):], _v)
