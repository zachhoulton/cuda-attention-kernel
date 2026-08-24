import torch


def main() -> int:
    batch = 2
    heads = 4
    seq_len = 8
    head_dim = 16

    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, heads, seq_len, head_dim)
    v = torch.randn(batch, heads, seq_len, head_dim)

    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)

    assert out.shape == q.shape
    print("Attention output shape is valid:", tuple(out.shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
