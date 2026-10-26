"""Convert original BF16 projections offline; retain BF16 source for other weights.

python -m startlux_prism.tools.convert_mxfp4 --model-dir ORIGINAL --out FP4_DIR
"""
import argparse
import json
from pathlib import Path

from safetensors.torch import save_file
import torch

from .. import hf_weights
from ..kernels import mxfp4 as F
from .. import mxfp4_artifact as A


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--components', choices=('gu', 'down', 'mlp', 'all'), default='all')
    p.add_argument('--scaling', choices=('ceil', 'mse'), default='ceil')
    p.add_argument('--rotate', action='store_true')
    a = p.parse_args()
    source, out = Path(a.model_dir), Path(a.out)
    if not torch.version.hip or not torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx950'):
        raise ValueError('This converter is validated on gfx950')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'mxfp4.json').exists():
        raise ValueError('Output already contains a complete artifact; choose a new directory')
    torch.set_num_threads(8)
    weights = hf_weights.load(source)
    manifest = dict(version=A.VERSION, layout=A.LAYOUT, components=a.components, scaling=a.scaling,
                    rotate=a.rotate, quantizer_sha256=A.quantizer_identity(),
                    source_config_sha256=A.sha(source / 'config.json'),
                    source_index_sha256=A.sha(source / 'model.safetensors.index.json'),
                    source='original BF16 safetensors; no GGUF requantization', layers=[])
    source_files = set(json.loads((source / 'model.safetensors.index.json').read_text())['weight_map'].values())
    manifest['source_shards'] = {name: dict(sha256=A.sha(source / name), bytes=(source / name).stat().st_size)
                                 for name in sorted(source_files)}
    names = {'gu', 'down'} if a.components == 'mlp' else {a.components}
    if a.components == 'all':
        names = {'gu', 'down', 'inp', 'out', 'qkv', 'o'}
    for i, layer in enumerate(weights.layers):
        matrices = dict(gu=(layer.ffn_gate, layer.ffn_up), down=(layer.ffn_down,))
        if weights.config.is_attention(i):
            matrices.update(qkv=(layer.wq, layer.wk, layer.wv), o=(layer.wo,))
        else:
            matrices.update(inp=(layer.wqkv, layer.wz, layer.wbeta, layer.walpha), out=(layer.wout,))
        payload, shapes = {}, {}
        for name in sorted(names & matrices.keys()):
            x = torch.cat([matrix.values for matrix in matrices[name]], 0).cuda()
            if not torch.isfinite(x).all():
                raise ValueError(f'Non-finite source weights at {i}:{name}')
            packed = F.quantize(x, a.scaling, a.rotate)
            payload[name + '.q'], payload[name + '.s'] = packed.q.cpu(), packed.s.cpu()
            shapes[name] = list(x.shape)
            del x, packed
        file = out / f'layer-{i:03d}.safetensors'
        tmp = file.with_suffix('.tmp')
        save_file(payload, str(tmp))
        tmp.replace(file)
        manifest['layers'].append(dict(file=file.name, sha256=A.sha(file), bytes=file.stat().st_size, shapes=shapes))
        print('Converted', i + 1, '/', len(weights.layers), flush=True)
    tmp = out / 'mxfp4.json.tmp'
    tmp.write_text(json.dumps(manifest, indent=2) + '\n')
    tmp.replace(out / 'mxfp4.json')
    print('Complete', sum(x['bytes'] for x in manifest['layers']), 'artifact bytes', flush=True)


if __name__ == '__main__':
    main()
