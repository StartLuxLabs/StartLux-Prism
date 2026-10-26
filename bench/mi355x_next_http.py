"""Matched HTTP comparison of previous INT8, next INT8 and native rotated FP4."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.request

from run_http import Client


def distribution(samples):
    """Linear-interpolated percentiles; preserve raw samples for paired analysis."""
    values = sorted(samples)
    def percentile(q):
        pos = (len(values) - 1) * q
        lo = int(pos)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo)
    return dict(n=len(values), p50_ms=percentile(.5), p95_ms=percentile(.95))


def summarize(measurements):
    result = {}
    for row in measurements:
        result.setdefault(row['profile'], {}).setdefault(row['suite'], []).append(row)
    return {profile: {suite: dict(distribution([x for row in rows for x in row['samples_ms']]),
                                 round_medians_ms=[row['median_ms'] for row in rows])
                      for suite, rows in suites.items()} for profile, suites in result.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', required=True, help='original BF16 model directory, also containing the INT8 GGUF')
    p.add_argument('--gguf', required=True, help='Q8_0 GGUF basename inside --model-dir')
    p.add_argument('--calib', required=True, help='INT8 calibration file; may be outside the checkout')
    p.add_argument('--mxfp4-artifact', help='converted FP4 directory; required for the mxfp4-r32 profile')
    p.add_argument('--manifest', required=True)
    p.add_argument('--image', default='startlux-prism:mi355x-next')
    p.add_argument('--candidate-image', help='compare a second image using the same FP4 recipe; requires --profiles mxfp4-r32')
    p.add_argument('--out', required=True)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--reps', type=int, default=100)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--port', type=int, default=18122)
    p.add_argument('--profiles', default='int8-stock,int8-next,mxfp4-r32')
    p.add_argument('--check-concurrency', action='store_true')
    a = p.parse_args()
    if min(a.rounds, a.reps, a.warmup) < 1:
        p.error('--rounds, --reps and --warmup must be positive')
    profiles = a.profiles.split(',')
    if len(set(profiles)) != len(profiles):
        p.error('--profiles must not contain duplicates')
    if any(profile not in ('int8-stock', 'int8-next', 'mxfp4-r32') for profile in profiles):
        p.error('--profiles must contain int8-stock, int8-next or mxfp4-r32')
    if 'mxfp4-r32' in profiles and not a.mxfp4_artifact:
        p.error('--mxfp4-artifact is required for the mxfp4-r32 profile')
    if a.candidate_image and profiles != ['mxfp4-r32']:
        p.error('--candidate-image requires --profiles mxfp4-r32')
    model, calib = Path(a.model_dir).resolve(), Path(a.calib).resolve()
    artifact = Path(a.mxfp4_artifact).resolve() if a.mxfp4_artifact else None
    if Path(a.gguf).name != a.gguf or not (model / a.gguf).is_file():
        p.error('--gguf must name an existing file inside --model-dir')
    if not calib.is_file():
        p.error('--calib must name an existing file')
    if 'mxfp4-r32' in profiles and not artifact.is_dir():
        p.error('--mxfp4-artifact must name an existing directory')
    output = Path(a.out).resolve()
    if output.exists():
        p.error('--out already exists; use a fresh run path')
    output.parent.mkdir(parents=True, exist_ok=True)
    bodies = json.loads(Path(a.manifest).read_text())['requests']
    images = {profile: a.image for profile in profiles}
    if a.candidate_image:
        profiles.append('mxfp4-candidate')
        images['mxfp4-candidate'] = a.candidate_image
    # Resolve tags once so an image rebuilt during a long run cannot change a later round.
    image_ids = {name: subprocess.check_output(['docker', 'image', 'inspect', tag, '--format', '{{.Id}}'],
                                              text=True).strip() for name, tag in images.items()}
    folder, gguf = '/models', '/models/' + a.gguf
    report = dict(image=a.image, image_ids=image_ids, rounds=a.rounds, warmup=a.warmup,
                  manifest_sha256=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  hip_visible_devices=os.getenv('HIP_VISIBLE_DEVICES', '0'),
                  measurements=[], health={}, responses={},
                  inputs=dict(model_dir=str(model), gguf=a.gguf, calib=str(calib),
                              mxfp4_artifact=str(artifact) if artifact else None))
    for round_no in range(a.rounds):
        order = profiles if round_no % 2 == 0 else list(reversed(profiles))
        for profile in order:
            fp4 = profile.startswith('mxfp4-')
            name = f'startlux-prism-mi355x-http-{os.getpid()}'
            command = ['docker', 'run', '--rm', '--name', name, '--network', 'host', '--ipc=host',
                       '--device=/dev/kfd', '--device=/dev/dri', '--group-add', 'video',
                       '--security-opt', 'seccomp=unconfined', '-e', 'HIP_VISIBLE_DEVICES=' + os.getenv('HIP_VISIBLE_DEVICES', '0'),
                       '-e', 'OMP_NUM_THREADS=8', '-v', str(model) + ':/models:ro',
                       '-v', str(calib) + ':/calib.pt:ro',
                       '-v', 'startlux-prism-mi355x-next-triton:/root/.triton']
            if fp4:
                command += ['-v', str(artifact) + ':/mxfp4:ro']
            if profile == 'int8-stock':
                for key, value in dict(INTRA_WARPS=8, HREC_BV=16, OUT_BV=32, HREC_NS=1, OUT_NS=1).items():
                    command += ['-e', f'STARTLUX_PRISM_{key}={value}']
            command += [image_ids[profile], '--gguf', folder if fp4 else gguf, '--model-dir', folder,
                        '--precision', 'mxfp4' if fp4 else 'int8', '--calib', '/calib.pt',
                        '--port', str(a.port), '--max-tokens', '4096', '--graphs', '--graph-max-tokens', '1536',
                        '--prefix-cache', '0', '--mmproj', 'none', '--think-mode', 'never',
                        '--max-think-budget', '64', '--think-seqs', '1', '--think-kv-rows', '4096']
            if fp4:
                command += ['--mxfp4-artifact', '/mxfp4']
            log_path = output.with_name(f'http-{round_no}-{profile}.log')
            with log_path.open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    base = 'http://127.0.0.1:' + str(a.port)
                    deadline = time.monotonic() + 900
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(f'{profile} exited: {log_path}')
                        try:
                            health = json.load(urllib.request.urlopen(base + '/health', timeout=2))
                            break
                        except OSError:
                            if time.monotonic() > deadline:
                                raise TimeoutError('server startup')
                            time.sleep(.25)
                    report['health'][profile] = health
                    print('READY', round_no, profile, flush=True)
                    client = Client(base + '/v1/systemone')
                    for suite in ('yesno', 'review1', 'readme3', 'startlux3', 'longpage2'):
                        body = json.dumps(bodies[suite]['body']).encode()
                        for _ in range(a.warmup):
                            _, reference = client.call(body)
                        stable = reference['answers'], reference['usage']
                        samples, timings, server_ms = [], [], []
                        for _ in range(a.reps):
                            ms, got = client.call(body)
                            assert (got['answers'], got['usage']) == stable, (profile, suite, 'unstable response')
                            samples.append(ms)
                            timings.append(got.get('timings', {}))
                            server_ms.append(got.get('latency_ms'))
                        old = report['responses'].setdefault(profile, {}).setdefault(suite, reference)
                        assert (old['answers'], old['usage']) == stable, (profile, suite, 'cross-round drift')
                        if profile == 'int8-next' and suite in report['responses'].get('int8-stock', {}):
                            prior = report['responses']['int8-stock'][suite]
                            assert (prior['answers'], prior['usage']) == stable, ('INT8 API output changed', suite)
                        peer = {'mxfp4-r32': 'mxfp4-candidate', 'mxfp4-candidate': 'mxfp4-r32'}.get(profile)
                        if peer and suite in report['responses'].get(peer, {}):
                            prior = report['responses'][peer][suite]
                            assert (prior['answers'], prior['usage']) == stable, ('FP4 candidate API output changed', suite)
                        row = dict(round=round_no, profile=profile, suite=suite, samples_ms=samples,
                                   median_ms=statistics.median(samples), timings=timings, server_ms=server_ms)
                        report['measurements'].append(row)
                        report['summary'] = summarize(report['measurements'])
                        output.write_text(json.dumps(report, indent=2) + '\n')
                        print(round_no, profile, suite, round(row['median_ms'], 3), 'ms', flush=True)
                    client.conn.close()
                    if a.check_concurrency:
                        check_path = output.with_name('concurrency-' + profile + '.json')
                        subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] /
                            'scripts/amd/test_mi355x_http.py'), '--url', base, '--output', str(check_path)], check=True)
                        report.setdefault('concurrency', {})[profile] = json.loads(check_path.read_text())
                        output.write_text(json.dumps(report, indent=2) + '\n')
                finally:
                    subprocess.run(['docker', 'stop', '--timeout', '5', name], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, check=False)
                    process.wait(timeout=30)


if __name__ == '__main__':
    main()
