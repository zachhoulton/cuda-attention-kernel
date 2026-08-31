#!/usr/bin/env python3
"""
Correctness test: Compare CUDA Flash-Attention backward pass (dQ/dK/dV) to
gradients produced by torch.autograd on the same reference forward pass.
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
    """Build and run the CUDA attention binary (produces forward + backward .bin files)."""
    repo_root = Path(__file__).parent.parent

    build_cmd = f"cd {repo_root} && bash scripts/build_flash_attention.sh"
    print(f"Building: {build_cmd}")
    result = subprocess.run(build_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Build failed:\n{result.stderr}")
    print(result.stdout)

    run_cmd = f"cd {repo_root} && bash scripts/run_flash_attention.sh"
    print(f"\nRunning: {run_cmd}")
    result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Execution failed:\n{result.stderr}")
    print(result.stdout)


def load_binary(filepath, shape):
    """Load binary output file and reshape to tensor."""
    with open(filepath, "rb") as f:
        data = np.fromfile(f, dtype=np.float32)
    return torch.from_numpy(data).reshape(shape)


def main():
    if not torch.cuda.is_available():
        print(
            "CUDA not available. This test must run on A100 with CUDA-enabled PyTorch."
        )
        return 1

    device = torch.device("cuda")

    # Config and deterministic inputs must match main() in csrc/flash_attention.cu
    batch, heads, seq_len, head_dim = 2, 8, 32, 64
    num_elements = batch * heads * seq_len * head_dim
    shape = (batch, heads, seq_len, head_dim)

    print("=== Backward Pass Correctness Test (dQ/dK/dV vs torch.autograd) ===\n")
    print(f"Configuration: batch={batch}, heads={heads}, seq_len={seq_len}, head_dim={head_dim}\n")

    q_vals = torch.tensor(
        [0.1 + 0.01 * (i % 100) for i in range(num_elements)],
        dtype=torch.float32, device=device
    ).reshape(shape)

    k_vals = torch.tensor(
        [0.2 + 0.02 * (i % 100) for i in range(num_elements)],
        dtype=torch.float32, device=device
    ).reshape(shape)

    v_vals = torch.tensor(
        [0.3 + 0.03 * (i % 100) for i in range(num_elements)],
        dtype=torch.float32, device=device
    ).reshape(shape)

    dout_vals = torch.tensor(
        [0.05 + 0.005 * (i % 100) for i in range(num_elements)],
        dtype=torch.float32, device=device
    ).reshape(shape)

    # Build and run the CUDA binary (writes dq/dk/dv_{noncausal,causal}.bin)
    print("=" * 60)
    run_cuda_binary()
    print("=" * 60)

    repo_root = Path(__file__).parent.parent
    build_dir = repo_root / "build"

    tol = 1e-4
    all_passed = True

    for causal in (False, True):
        label = "causal" if causal else "noncausal"
        print(f"\n--- {label} ---")

        q = q_vals.clone().requires_grad_(True)
        k = k_vals.clone().requires_grad_(True)
        v = v_vals.clone().requires_grad_(True)

        out = reference_attention(q, k, v, causal=causal)
        out.backward(gradient=dout_vals)

        for name, ref_grad in (("dq", q.grad), ("dk", k.grad), ("dv", v.grad)):
            path = build_dir / f"{name}_{label}.bin"
            if not path.exists():
                print(f"✗ Error: {path} not found")
                all_passed = False
                continue

            cuda_grad = load_binary(path, shape).to(device)
            match = torch.allclose(ref_grad, cuda_grad, atol=tol, rtol=tol)
            if match:
                print(f"✓ {name} matches PyTorch autograd (tol={tol:.2e})")
            else:
                diff = (ref_grad - cuda_grad).abs().max().item()
                print(f"✗ {name} MISMATCH (max diff={diff:.2e})")
                all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("\n✓ All backward-pass tests passed!")
        return 0
    else:
        print("\n✗ Some backward-pass tests failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
