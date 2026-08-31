#!/usr/bin/env python3
"""
Correctness test: reduced-precision (fp16/bf16) forward+backward vs a PyTorch reference.
"""

import math
import subprocess
from pathlib import Path

import numpy as np
import torch


def reference_attention(q, k, v, causal=False):
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
    with open(filepath, "rb") as f:
        data = np.fromfile(f, dtype=np.float32)
    return torch.from_numpy(data).reshape(shape)


# (torch dtype, CUDA-side file label, comparison tolerance)
DTYPE_CONFIGS = [
    (torch.float16, "fp16", 5e-2),
    (torch.bfloat16, "bf16", 5e-2),
]


def main():
    if not torch.cuda.is_available():
        print("CUDA not available. This test must run on A100 with CUDA-enabled PyTorch.")
        return 1

    device = torch.device("cuda")
    batch, heads, seq_len, head_dim = 2, 8, 32, 64
    num_elements = batch * heads * seq_len * head_dim
    shape = (batch, heads, seq_len, head_dim)

    print("=== Reduced-Precision Correctness Test (fp16/bf16) ===\n")

    q_f32 = torch.tensor(
        [0.1 + 0.01 * (i % 100) for i in range(num_elements)],
        dtype=torch.float32, device=device
    ).reshape(shape)
    k_f32 = torch.tensor(
        [0.2 + 0.02 * (i % 100) for i in range(num_elements)],
        dtype=torch.float32, device=device
    ).reshape(shape)
    v_f32 = torch.tensor(
        [0.3 + 0.03 * (i % 100) for i in range(num_elements)],
        dtype=torch.float32, device=device
    ).reshape(shape)
    dout_f32 = torch.tensor(
        [0.05 + 0.005 * (i % 100) for i in range(num_elements)],
        dtype=torch.float32, device=device
    ).reshape(shape)

    print("=" * 60)
    run_cuda_binary()
    print("=" * 60)

    repo_root = Path(__file__).parent.parent
    build_dir = repo_root / "build"

    all_passed = True

    for torch_dtype, label, tol in DTYPE_CONFIGS:
        print(f"\n--- {label} (tol={tol:.1e}) ---")

        # Round inputs through the target dtype, matching what the kernel actually reads
        q = q_f32.to(torch_dtype).float().requires_grad_(True)
        k = k_f32.to(torch_dtype).float().requires_grad_(True)
        v = v_f32.to(torch_dtype).float().requires_grad_(True)
        dout = dout_f32.to(torch_dtype).float()

        out = reference_attention(q, k, v, causal=False)
        out.backward(gradient=dout)

        checks = [
            ("out", out.detach()),
            ("dq", q.grad),
            ("dk", k.grad),
            ("dv", v.grad),
        ]

        for name, ref in checks:
            file_prefix = "attention_noncausal" if name == "out" else f"{name}_noncausal"
            path = build_dir / f"{file_prefix}_{label}.bin"
            if not path.exists():
                print(f"✗ Error: {path} not found")
                all_passed = False
                continue

            cuda_val = load_binary(path, shape).to(device)
            match = torch.allclose(ref, cuda_val, atol=tol, rtol=tol)
            if match:
                print(f"✓ {name} matches PyTorch reference")
            else:
                diff = (ref - cuda_val).abs().max().item()
                print(f"✗ {name} MISMATCH (max diff={diff:.2e})")
                all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("\n✓ All reduced-precision tests passed!")
        return 0
    else:
        print("\n✗ Some reduced-precision tests failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
