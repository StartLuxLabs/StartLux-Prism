"""Exercise Prism's RCCL TP path on real GPUs, including HIP graph replay.

HIP_VISIBLE_DEVICES=0,2 torchrun --standalone --nproc-per-node=2 scripts/amd/test_mi355x_collectives.py
"""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.distributed as dist
from startlux_prism import tp as TP
from startlux_prism.kernels import quant as Q


def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    port = int(os.environ["MASTER_PORT"])
    group = TP.Group(rank, world, port, f"startlux-prism-rccl-test-{port}")
    try:
        group.setup(256, 4)
        group.begin(4)
        assert not group.fast and not group.pull
        host = group.shared("staging", 64)
        host.copy_(torch.arange(64, dtype=torch.uint8))
        assert torch.equal(host.cuda().cpu(), host)

        x = torch.arange(512, device=group.device, dtype=torch.float32).view(4, 128) * (rank + 1)
        q = torch.empty_like(x, dtype=torch.int8)
        scales = torch.empty(4, device=group.device)
        maxima = torch.empty_like(scales)
        live = torch.tensor([4], device=group.device, dtype=torch.int32)
        inv = torch.ones(128, device=group.device)

        def run():
            TP.quantize_rows(group, Q.rows_q8, 4, maxima, x, inv, q, scales, live, 4)

        run()
        expected_scale = (x / (rank + 1) * world).abs().amax(1) / 127
        assert torch.equal(scales, expected_scale)
        assert torch.equal(q, torch.round(x / expected_scale[:, None]).to(torch.int8))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        graph.replay()
        assert torch.equal(scales, expected_scale)
        del graph
        sums = torch.full((17, 65), rank + 1, dtype=torch.int32, device=group.device)
        group.sum_(sums)
        assert torch.all(sums == world * (world + 1) // 2)
        dist.barrier()
        print(f"PASS rank {rank}: shared staging, exact global quantization, HIP graph replay, int32 RCCL sums", flush=True)
    finally:
        dist.destroy_process_group()
        TP._unlink_created()


if __name__ == "__main__":
    main()
