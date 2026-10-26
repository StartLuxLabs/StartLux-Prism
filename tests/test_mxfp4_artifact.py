"""Artifact integrity and source/recipe mismatches must fail before model execution."""
import json

import pytest
import torch
from safetensors.torch import save_file

from startlux_prism import mxfp4_artifact as A


def test_manifest_rejects_source_and_recipe_changes(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'config.json').write_text('{}')
    (source / 'model.safetensors.index.json').write_text('{"weight_map":{"w":"model.safetensors"}}')
    (source / 'model.safetensors').write_bytes(b'original weights')
    m = dict(version=A.VERSION, layout=A.LAYOUT, components='all', scaling='ceil', rotate=True,
             quantizer_sha256=A.quantizer_identity(), source_config_sha256=A.sha(source / 'config.json'),
             source_index_sha256=A.sha(source / 'model.safetensors.index.json'), layers=[],
             source_shards={'model.safetensors': {'sha256': A.sha(source / 'model.safetensors')}})
    (tmp_path / 'mxfp4.json').write_text(json.dumps(m))
    assert A.read_manifest(tmp_path, source, 'all', 'ceil', True) == m
    with pytest.raises(ValueError, match='scaling'):
        A.read_manifest(tmp_path, source, 'all', 'mse', True)
    (source / 'config.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='source_config'):
        A.read_manifest(tmp_path, source, 'all', 'ceil', True)
    (source / 'config.json').write_text('{}')
    (source / 'model.safetensors').write_bytes(b'different weights')
    with pytest.raises(ValueError, match='source shard differs'):
        A.read_manifest(tmp_path, source, 'all', 'ceil', True)


def test_payload_checksum_and_shape_validation(tmp_path):
    p = tmp_path / 'layer.safetensors'
    q, s = torch.zeros((1, 16), dtype=torch.uint8), torch.full((1, 1), 127, dtype=torch.uint8)
    save_file({'gu.q': q, 'gu.s': s}, str(p))
    entry = dict(file=p.name, sha256=A.sha(p), shapes={'gu': [1, 32]})
    got = A.load_layer(tmp_path, entry, 'cpu')['gu']
    assert torch.equal(got.q, q) and torch.equal(got.s, s)
    with pytest.raises(ValueError, match='checksum'):
        A.load_layer(tmp_path, entry | {'sha256': '0' * 64}, 'cpu')
    with pytest.raises(ValueError, match='Malformed'):
        A.load_layer(tmp_path, entry | {'shapes': {'gu': [2, 32]}}, 'cpu')
    with pytest.raises(ValueError, match='path/checksum'):
        A.load_layer(tmp_path, entry | {'file': '../elsewhere'}, 'cpu')
