"""Freeze source-model identities and exact HTTP request bodies before FP4 trials."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

from run_http import make_body
import suites


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf', required=True)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--calib', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    model = Path(a.model_dir)
    repo = 'startlux-models/StartLux-Decision-v1.1-27B-AutoThink'
    revision = 'dd62d5cd84a483d75bdd548880bbf08bdd3f67a4'
    api = f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true'
    upstream = json.load(urllib.request.urlopen(api))
    manifest = dict(baseline='4534c11', source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(), original_model=dict(repo=repo, revision=revision),
        options=dict(prefix_cache=0, graphs=True, think_mode='never', concurrency=1), files={}, requests={})
    files = [Path(a.gguf), Path(a.calib)] + [model / name for name in (
        'config.json', 'decision_config.json', 'tokenizer.json', 'tokenizer_config.json',
        'chat_template.jinja', 'model.safetensors.index.json')]
    for path in files:
        manifest['files'][path.name] = dict(bytes=path.stat().st_size, sha256=sha(path))
    manifest['original_model']['shards'] = {
        x['rfilename']: dict(bytes=x['lfs']['size'], sha256=x['lfs']['sha256'])
        for x in upstream['siblings'] if x['rfilename'].endswith('.safetensors')}
    for name, (state, questions) in suites.fixed().items():
        body = json.loads(make_body(state, questions, unique=False))
        body['think_mode'] = 'never'
        encoded = json.dumps(body, sort_keys=True, separators=(',', ':')).encode()
        manifest['requests'][name] = dict(body=body, sha256=hashlib.sha256(encoded).hexdigest())
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print('Frozen', len(files), 'files,', len(manifest['requests']), 'requests and original shard identities')


if __name__ == '__main__':
    main()
