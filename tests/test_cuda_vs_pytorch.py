#!/usr/bin/env python3
"""
Correctness test: Compare CUDA Flash-Attention output to PyTorch reference.
"""

import math
import re
import subprocess
from pathlib import Path

import torch


def reference_attention(q, k, v, causal=False):
    """Compute attention using standard PyTorch operations."""
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if causal:
        seq_len = q.size(-2)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def parse_cuda_output(output_str, label):
    """Extract matrix from CUDA output between markers."""
    start_marker = f"TEST_{label}_OUTPUT_START"
    end_marker = f"TEST_{label}_OUTPUT_END"

    start_idx = output_str.find(start_marker)
    end_idx = output_str.find(end_marker)

    if start_idx == -1 or end_idx == -1:
        raise ValueError(f"Markers for {label} not found in CUDA output")

    output_section = output_str[start_idx + len(start_marker) : end_idx]
    values = [float(v) for v in output_section.split() if v]

    # Reshape to (seq_len, head_dim) = (8, 16)
    seq_len, head_dim = 8, 16
    result = torch.tensor(values, dtype=torch.float32).reshape(seq_len, head_dim)
    return result


def run_cuda_binary():
    """Build and run the CUDA attention binary."""
    repo_root = Path(__file__).parent.parent

    # Build
    build_cmd = f"cd {repo_root} && bash scripts/build_flash_attention.sh"
    print(f"Building: {build_cmd}")
    result = subprocess.run(build_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Build failed: {result.stderr}")

    # Run
    run_cmd = f"cd {repo_root} && bash scripts/run_flash_attention.sh"
    print(f"Running: {run_cmd}")
    result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Execution failed: {result.stderr}")

    return result.stdout


def main():
    if not torch.cuda.is_available():
        print(
            "CUDA not available. This test must run on A100 with CUDA-enabled PyTorch."
        )
        return 1

    device = torch.device("cuda")

    # Generate deterministic inputs
    seq_len, head_dim = 8, 16
    num_elements = seq_len * head_dim

    # Initialize q, k, v with same logic as C++ main
    q_vals = [0.1 + 0.01 * (i % 100) for i in range(num_elements)]
    k_vals = [0.2 + 0.02 * (i % 100) for i in range(num_elements)]
    v_vals = [0.3 + 0.03 * (i % 100) for i in range(num_elements)]

    q = torch.tensor(q_vals, dtype=torch.float32, device=device).reshape(1, 1, seq_len, head_dim)
    k = torch.tensor(k_vals, dtype=torch.float32, device=device).reshape(1, 1, seq_len, head_dim)
    v = torch.tensor(v_vals, dtype=torch.float32, device=device).reshape(1, 1, seq_len, head_dim)

    # Compute PyTorch reference
    print("\n=== Computing PyTorch Reference ===")
    ref_noncausal = reference_attention(q, k, v, causal=False)
    ref_causal = reference_attention(q, k, v, causal=True)

    print("PyTorch non-causal output shape:", ref_noncausal.shape)
    print("PyTorch non-causal output[0]:", ref_noncausal[0, 0, 0, :4])

    print("PyTorch causal output shape:", ref_causal.shape)
    print("PyTorch causal output[0]:", ref_causal[0, 0, 0, :4])

    # Run CUDA binary and parse output
    print("\n=== Running CUDA Kernel ===")
    cuda_output = run_cuda_binary()

    cuda_noncausal = parse_cuda_output(cuda_output, "NONCAUSAL").unsqueeze(0).unsqueeze(0).to(device)
    cuda_causal = parse_cuda_output(cuda_output, "CAUSAL").unsqueeze(0).unsqueeze(0).to(device)

    print("CUDA non-causal output[0]:", cuda_noncausal[0, 0, 0, :4])
    print("CUDA causal output[0]:", cuda_causal[0, 0, 0, :4])

    # Compare with tolerance
    print("\n=== Correctness Validation ===")
    tol = 1e-4
    
    noncausal_match = torch.allclose(ref_noncausal, cuda_noncausal, atol=tol, rtol=tol)
    causal_match = torch.allclose(ref_causal, cuda_causal, atol=tol, rtol=tol)

    if noncausal_match:
        print("✓ Non-causal attention matches PyTorch reference (tol=%.2e)" % tol)
    else:
        diff = (ref_noncausal - cuda_noncausal).abs().max().item()
        print(f"✗ Non-causal attention MISMATCH (max diff={diff:.2e})")
        print("  Expected (first 8):", ref_noncausal[0, 0, 0, :8])
        print("  Got (first 8):     ", cuda_noncausal[0, 0, 0, :8])

    if causal_match:
        print("✓ Causal attention matches PyTorch reference (tol=%.2e)" % tol)
    else:
        diff = (ref_causal - cuda_causal).abs().max().item()
        print(f"✗ Causal attention MISMATCH (max diff={diff:.2e})")
        print("  Expected (first 8):", ref_causal[0, 0, 0, :8])
        print("  Got (first 8):     ", cuda_causal[0, 0, 0, :8])

    if noncausal_match and causal_match:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
