#!/usr/bin/env python3
"""
Comparison: CUDA flash-attention forward+backward kernels vs both
torch.nn.functional.scaled_dot_product_attention and naive PyTorch attention
"""

import math
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


def naive_attention(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def benchmark_naive_forward(seq_len):
    device = torch.device("cuda")
    q = torch.randn(BATCH, HEADS, seq_len, HEAD_DIM, device=device, dtype=torch.float32)
    k = torch.randn(BATCH, HEADS, seq_len, HEAD_DIM, device=device, dtype=torch.float32)
    v = torch.randn(BATCH, HEADS, seq_len, HEAD_DIM, device=device, dtype=torch.float32)

    for _ in range(WARMUP_ITERS):
        naive_attention(q, k, v)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIMED_ITERS):
        naive_attention(q, k, v)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / TIMED_ITERS


def benchmark_naive_backward(seq_len):
    device = torch.device("cuda")
    q = torch.randn(BATCH, HEADS, seq_len, HEAD_DIM, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(BATCH, HEADS, seq_len, HEAD_DIM, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(BATCH, HEADS, seq_len, HEAD_DIM, device=device, dtype=torch.float32, requires_grad=True)

    out = naive_attention(q, k, v)
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


def print_comparison(title, other_label, custom_results, other_results):
    print(f"\n=== {title}: custom vs {other_label} ===")
    header = f"{'seq_len':<10} {'custom(ms)':<12} {other_label + '(ms)':<12} {'custom GFLOP/s':<16} {other_label + ' GFLOP/s':<16} {'winner':<20}"
    print(header)
    for seq_len in SEQ_LENS:
        c_ms, c_gflops = custom_results.get(seq_len, (float("nan"), float("nan")))
        o_ms, o_gflops = other_results[seq_len]
        if c_ms < o_ms:
            winner = f"custom {o_ms / c_ms:.2f}x faster"
        else:
            winner = f"{other_label} {c_ms / o_ms:.2f}x faster"
        print(f"{seq_len:<10} {c_ms:<12.4f} {o_ms:<12.4f} {c_gflops:<16.2f} {o_gflops:<16.2f} {winner:<20}")


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

    print("\n=== naive PyTorch ===")
    print(f"{'seq_len':<10} {'latency(ms)':<12} {'GFLOP/s':<12}")
    naive_forward = {}
    for seq_len in SEQ_LENS:
        latency_ms = benchmark_naive_forward(seq_len)
        gflops = forward_flops(seq_len) / (latency_ms / 1000.0) / 1e9
        naive_forward[seq_len] = (latency_ms, gflops)
        print(f"{seq_len:<10} {latency_ms:<12.4f} {gflops:<12.2f}")

    print("\n=== naive PyTorch: backward ===")
    print(f"{'seq_len':<10} {'latency(ms)':<12} {'GFLOP/s':<12}")
    naive_backward = {}
    for seq_len in SEQ_LENS:
        latency_ms = benchmark_naive_backward(seq_len)
        gflops = backward_flops(seq_len) / (latency_ms / 1000.0) / 1e9
        naive_backward[seq_len] = (latency_ms, gflops)
        print(f"{seq_len:<10} {latency_ms:<12.4f} {gflops:<12.2f}")

    print_comparison("Forward", "sdpa", custom_forward, sdpa_forward)
    print_comparison("Forward", "naive", custom_forward, naive_forward)
    print_comparison("Backward", "sdpa", custom_backward, sdpa_backward)
    print_comparison("Backward", "naive", custom_backward, naive_backward)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

