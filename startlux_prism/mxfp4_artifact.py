"""Versioned, checksummed FP4 projection artifacts; source BF16 files stay separate."""
import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file

from .kernels import mxfp4 as F

VERSION = 1
LAYOUT = 'e2m1-adjacent-k/e8m0-per-32-k'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def quantizer_identity():
    return hashlib.sha256(''.join(fn.src for fn in (F._quant, F._encode, F._magnitude, F._hadamard32)).encode()).hexdigest()


def read_manifest(root, source, components, scaling, rotate):
    root, source = Path(root), Path(source)
    manifest = json.loads((root / 'mxfp4.json').read_text())
    expected = dict(version=VERSION, layout=LAYOUT, components=components, scaling=scaling, rotate=rotate,
                    quantizer_sha256=quantizer_identity(), source_config_sha256=sha(source / 'config.json'),
                    source_index_sha256=sha(source / 'model.safetensors.index.json'))
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f'MXFP4 artifact {key} does not match this source/profile/quantizer')
    files = set(json.loads((source / 'model.safetensors.index.json').read_text())['weight_map'].values())
    if files != set(manifest.get('source_shards', {})):
        raise ValueError('MXFP4 artifact source shard identities are missing or incomplete')
    source = source.resolve()
    for name in sorted(files):
        path = (source / name).resolve()
        if not path.is_relative_to(source) or sha(path) != manifest['source_shards'][name]['sha256']:
            raise ValueError(f'MXFP4 artifact source shard differs: {name}')
    return manifest


def load_layer(root, entry, device):
    root = Path(root).resolve()
    file = (root / entry['file']).resolve()
    if not file.is_relative_to(root) or sha(file) != entry['sha256']:
        raise ValueError('MXFP4 artifact path/checksum is invalid')
    raw = load_file(str(file))
    packed = {}
    for name, shape in entry['shapes'].items():
        q, s = raw[name + '.q'], raw[name + '.s']
        n, k = shape
        if str(q.dtype) != 'torch.uint8' or str(s.dtype) != 'torch.uint8' or q.shape != (n, k // 2) or s.shape != (n, k // 32):
            raise ValueError(f'Malformed MXFP4 payload/scales for {name}')
        packed[name] = F.prepare(F.Packed(q.to(device), s.to(device)))
    return packed
