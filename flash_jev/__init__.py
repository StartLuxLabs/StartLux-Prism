"""The former name of startlux_prism (StartLux-Prism). `import flash_jev.…` and `python -m flash_jev.server` (or .dp,
.calib, .tp) keep working: this package reads its modules from startlux_prism's directory."""
import startlux_prism as _prism

__path__ = _prism.__path__
