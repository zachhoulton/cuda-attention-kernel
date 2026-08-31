#!/usr/bin/env python3
"""
Correctness test: Compare CUDA Flash-Attention output to PyTorch reference.
"""

import math
import numpy as np
import subprocess
from pathlib import Path

import torch


def reference_attention(q, k, v, causal=False):
    """Compute attention using standard PyTorch operations.
    
    Shape: q, k, v are (batch, heads, seq_len, head_dim)
    Returns: (batch, heads, seq_len, head_dim)
    """
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


def run_cuda_binary():
    """Build and run the CUDA attention binary."""
    repo_root = Path(__file__).parent.parent

    # Build
    build_cmd = f"cd {repo_root} && bash scripts/build_flash_attention.sh"
    print(f"Building: {build_cmd}")
    result = subprocess.run(build_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Build failed:\n{result.stderr}")
    print(result.stdout)

    # Run
    run_cmd = f"cd {repo_root} && bash scripts/run_flash_attention.sh"
    print(f"\nRunning: {run_cmd}")
    result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Execution failed:\n{result.stderr}")
    print(result.stdout)


def load_binary_output(filepath, shape):
    """Load binary output file and reshape to tensor."""
    with open(filepath, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    return torch.from_numpy(data).reshape(shape)


def main():
    if not torch.cuda.is_available():
        print(
            "CUDA not available. This test must run on A100 with CUDA-enabled PyTorch."
        )
        return 1

    device = torch.device("cuda")

    # Config must match C++ kernel test
    batch, heads, seq_len, head_dim = 2, 8, 32, 64
    num_elements = batch * heads * seq_len * head_dim

    print("=== Multi-head + Batch Attention Correctness Test ===\n")
    print(f"Configuration: batch={batch}, heads={heads}, seq_len={seq_len}, head_dim={head_dim}")
    print(f"Total elements: {num_elements}\n")

    # Generate deterministic inputs
    q_vals = torch.tensor(
        [0.1 + 0.01 * (i % 100) for i in range(num_elements)],
        dtype=torch.float32,
        device=device
    ).reshape(batch, heads, seq_len, head_dim)
    
    k_vals = torch.tensor(
        [0.2 + 0.02 * (i % 100) for i in range(num_elements)],
        dtype=torch.float32,
        device=device
    ).reshape(batch, heads, seq_len, head_dim)
    
    v_vals = torch.tensor(
        [0.3 + 0.03 * (i % 100) for i in range(num_elements)],
        dtype=torch.float32,
        device=device
    ).reshape(batch, heads, seq_len, head_dim)

    # Compute PyTorch reference
    print("Computing PyTorch reference attention...\n")
    ref_noncausal = reference_attention(q_vals, k_vals, v_vals, causal=False)
    ref_causal = reference_attention(q_vals, k_vals, v_vals, causal=True)

    print(f"PyTorch non-causal output shape: {ref_noncausal.shape}")
    print(f"  Sample (batch=0, head=0, tokens 0-3):")
    for i in range(4):
        print(f"    Token {i}: {ref_noncausal[0, 0, i, :4]}")

    print(f"\nPyTorch causal output shape: {ref_causal.shape}")
    print(f"  Sample (batch=0, head=0, tokens 0-3):")
    for i in range(4):
        print(f"    Token {i}: {ref_causal[0, 0, i, :4]}")

    # Build and run CUDA kernel
    print("\n" + "="*60)
    run_cuda_binary()

    # Load CUDA outputs
    print("\n" + "="*60)
    print("\nLoading CUDA outputs...\n")
    
    repo_root = Path(__file__).parent.parent
    noncausal_file = repo_root / "build" / "attention_noncausal.bin"
    causal_file = repo_root / "build" / "attention_causal.bin"

    if not noncausal_file.exists():
        print(f"✗ Error: {noncausal_file} not found")
        return 1
    if not causal_file.exists():
        print(f"✗ Error: {causal_file} not found")
        return 1

    cuda_noncausal = load_binary_output(noncausal_file, (batch, heads, seq_len, head_dim)).to(device)
    cuda_causal = load_binary_output(causal_file, (batch, heads, seq_len, head_dim)).to(device)

    print(f"CUDA non-causal output shape: {cuda_noncausal.shape}")
    print(f"  Sample (batch=0, head=0, tokens 0-3):")
    for i in range(4):
        print(f"    Token {i}: {cuda_noncausal[0, 0, i, :4]}")

    print(f"\nCUDA causal output shape: {cuda_causal.shape}")
    print(f"  Sample (batch=0, head=0, tokens 0-3):")
    for i in range(4):
        print(f"    Token {i}: {cuda_causal[0, 0, i, :4]}")

    # Compare with tolerance
    print("\n" + "="*60)
    print("\n=== Correctness Validation ===\n")
    
    tol = 1e-4

    noncausal_match = torch.allclose(ref_noncausal, cuda_noncausal, atol=tol, rtol=tol)
    if noncausal_match:
        print(f"✓ Non-causal attention matches PyTorch reference (tol={tol:.2e})")
    else:
        diff = (ref_noncausal - cuda_noncausal).abs().max().item()
        print(f"✗ Non-causal attention MISMATCH (max diff={diff:.2e})")

    causal_match = torch.allclose(ref_causal, cuda_causal, atol=tol, rtol=tol)
    if causal_match:
        print(f"✓ Causal attention matches PyTorch reference (tol={tol:.2e})")
    else:
        diff = (ref_causal - cuda_causal).abs().max().item()
        print(f"✗ Causal attention MISMATCH (max diff={diff:.2e})")

    if noncausal_match and causal_match:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())