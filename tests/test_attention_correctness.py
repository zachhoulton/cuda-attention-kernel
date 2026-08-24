import math

import torch


def reference_attention(q, k, v, causal: bool = False):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if causal:
        seq_len = q.size(-2)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install a CUDA-enabled PyTorch build and NVIDIA GPU drivers before running the attention correctness test."
        )

    device = torch.device("cuda")
    q = torch.randn(1, 1, 8, 16, device=device, dtype=torch.float32)
    k = torch.randn(1, 1, 8, 16, device=device, dtype=torch.float32)
    v = torch.randn(1, 1, 8, 16, device=device, dtype=torch.float32)

    ref = reference_attention(q, k, v, causal=False)
    print("Reference attention output shape:", tuple(ref.shape))
    print("Reference output sample:", ref[0, 0, 0, :4])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
