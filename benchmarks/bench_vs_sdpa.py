#!/usr/bin/env python3
"""
Comparison: CUDA flash-attention forward+backward kernels vs
torch.nn.functional.scaled_dot_product_attention
"""

import re
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

WARMUP_ITERS = 5
TIMED_ITERS = 20
SEQ_LENS = [128, 256, 512, 1024, 2048, 4096]
BATCH, HEADS, HEAD_DIM = 2, 8, 64

REPO_ROOT = Path(__file__).parent.parent
RESULT_LINE_RE = re.compile(r"^(\d+)\s+([\d.]+)\s+([\d.]+)\s*$")


def forward_flops(seq_len):
    return 4.0 * BATCH * HEADS * seq_len * seq_len * HEAD_DIM


# Backward does ~2.5x forward's FLOPs (standard convention)
def backward_flops(seq_len):
    return 10.0 * BATCH * HEADS * seq_len * seq_len * HEAD_DIM


def run_custom_kernel():
    """Build and run the CUDA benchmark binary, parsing its Forward/Backward seq_len tables."""
    subprocess.run("bash scripts/build_benchmark.sh", shell=True, cwd=REPO_ROOT, check=True)
    result = subprocess.run(
        "bash scripts/run_benchmark.sh", shell=True, cwd=REPO_ROOT,
        check=True, capture_output=True, text=True,
    )
    print(result.stdout)

    forward_results, backward_results = {}, {}
    current = None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped == "=== Forward ===":
            current = forward_results
        elif stripped == "=== Backward ===":
            current = backward_results
        elif current is not None:
            match = RESULT_LINE_RE.match(stripped)
            if match:
                seq_len, latency_ms, gflops = match.groups()
                current[int(seq_len)] = (float(latency_ms), float(gflops))
    return forward_results, backward_results


def benchmark_sdpa_forward(seq_len):
    device = torch.device("cuda")
    q = torch.randn(BATCH, HEADS, seq_len, HEAD_DIM, device=device, dtype=torch.float32)
    k = torch.randn(BATCH, HEADS, seq_len, HEAD_DIM, device=device, dtype=torch.float32)
    v = torch.randn(BATCH, HEADS, seq_len, HEAD_DIM, device=device, dtype=torch.float32)

    for _ in range(WARMUP_ITERS):
        F.scaled_dot_product_attention(q, k, v, is_causal=False)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIMED_ITERS):
        F.scaled_dot_product_attention(q, k, v, is_causal=False)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / TIMED_ITERS


def benchmark_sdpa_backward(seq_len):
    device = torch.device("cuda")
    q = torch.randn(BATCH, HEADS, seq_len, HEAD_DIM, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(BATCH, HEADS, seq_len, HEAD_DIM, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(BATCH, HEADS, seq_len, HEAD_DIM, device=device, dtype=torch.float32, requires_grad=True)

    out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    grad_output = torch.randn_like(out)

    for _ in range(WARMUP_ITERS):
        torch.autograd.grad(out, (q, k, v), grad_outputs=grad_output, retain_graph=True)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIMED_ITERS):
        torch.autograd.grad(out, (q, k, v), grad_outputs=grad_output, retain_graph=True)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / TIMED_ITERS


def print_comparison(title, custom_results, sdpa_results):
    print(f"\n=== {title} comparison ===")
    header = f"{'seq_len':<10} {'custom(ms)':<12} {'sdpa(ms)':<12} {'custom GFLOP/s':<16} {'sdpa GFLOP/s':<14} {'sdpa speedup':<12}"
    print(header)
    for seq_len in SEQ_LENS:
        c_ms, c_gflops = custom_results.get(seq_len, (float("nan"), float("nan")))
        s_ms, s_gflops = sdpa_results[seq_len]
        speedup = c_ms / s_ms if s_ms else float("nan")
        print(f"{seq_len:<10} {c_ms:<12.4f} {s_ms:<12.4f} {c_gflops:<16.2f} {s_gflops:<14.2f} {speedup:<11.2f}x")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available. This benchmark must run on a CUDA-enabled machine.")
        return 1

    print(f"Configuration: batch={BATCH}, heads={HEADS}, head_dim={HEAD_DIM}, non-causal fp32\n")

    print("=== Custom CUDA kernel ===")
    custom_forward, custom_backward = run_custom_kernel()

    print("=== torch.nn.functional.scaled_dot_product_attention: forward ===")
    print(f"{'seq_len':<10} {'latency(ms)':<12} {'GFLOP/s':<12}")
    sdpa_forward = {}
    for seq_len in SEQ_LENS:
        latency_ms = benchmark_sdpa_forward(seq_len)
        gflops = forward_flops(seq_len) / (latency_ms / 1000.0) / 1e9
        sdpa_forward[seq_len] = (latency_ms, gflops)
        print(f"{seq_len:<10} {latency_ms:<12.4f} {gflops:<12.2f}")

    print("\n=== torch.nn.functional.scaled_dot_product_attention: backward ===")
    print(f"{'seq_len':<10} {'latency(ms)':<12} {'GFLOP/s':<12}")
    sdpa_backward = {}
    for seq_len in SEQ_LENS:
        latency_ms = benchmark_sdpa_backward(seq_len)
        gflops = backward_flops(seq_len) / (latency_ms / 1000.0) / 1e9
        sdpa_backward[seq_len] = (latency_ms, gflops)
        print(f"{seq_len:<10} {latency_ms:<12.4f} {gflops:<12.2f}")

    print_comparison("Forward", custom_forward, sdpa_forward)
    print_comparison("Backward", custom_backward, sdpa_backward)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

