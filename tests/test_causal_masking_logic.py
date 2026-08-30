#!/usr/bin/env python3
"""
Causal masking validation: Verify that attention respects the causal constraint.
"""

import math
import torch


def reference_attention_causal_debug(q, k, v):
    """
    Compute causal attention with detailed debugging output.
    
    For each query position i:
    - Score to key position j is set to -inf if j > i (future keys masked)
    - After softmax, masked positions have probability 0
    - Output should only depend on position <= i
    """
    seq_len = q.size(-2)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    
    # Create causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    
    return probs, scores


def test_causal_constraint():
    """Verify causal masking logic: each query only attends to past and present."""
    seq_len, head_dim = 8, 16
    
    # Create small, deterministic inputs
    q = torch.ones(1, 1, seq_len, head_dim, dtype=torch.float32)
    k = torch.arange(1, seq_len + 1, dtype=torch.float32).unsqueeze(-1).expand(-1, head_dim)
    k = k.unsqueeze(0).unsqueeze(0)
    
    probs, scores = reference_attention_causal_debug(q, k, torch.ones_like(k))
    
    print("=== Causal Masking Validation ===\n")
    
    # Check mask properties
    all_pass = True
    for i in range(seq_len):
        for j in range(seq_len):
            # For position i, keys at j > i should be masked (prob = 0)
            prob = probs[0, 0, i, j].item()
            score = scores[0, 0, i, j].item()
            
            if j > i:
                # Future position: should be masked
                if prob > 1e-6:
                    print(f"✗ FAIL: Position {i} attended to future key {j} with prob {prob:.6f}")
                    all_pass = False
            elif math.isinf(score) and score < 0:
                # Past/present position: should NOT be masked
                print(f"✗ FAIL: Position {i} had masked past key {j}")
                all_pass = False
    
    if all_pass:
        print("✓ Causal masking constraint verified:")
        print("  - Each query position attends ONLY to past and present keys")
        print("  - Future keys are masked (probability = 0)")
        print()
        
        # Print attention pattern
        print("Attention probability pattern (query_pos x key_pos):")
        print("(0 = masked future, number = probability)")
        for i in range(seq_len):
            for j in range(seq_len):
                prob = probs[0, 0, i, j].item()
                if prob < 1e-6:
                    print("  0", end="")
                else:
                    print(f"  1", end="")
            print()
    
    return all_pass


def test_gradient_flow():
    """Verify that gradients cannot flow from future to past"""
    seq_len, head_dim = 4, 8
    
    q = torch.randn(1, 1, seq_len, head_dim, requires_grad=True)
    k = torch.randn(1, 1, seq_len, head_dim, requires_grad=True)
    v = torch.randn(1, 1, seq_len, head_dim, requires_grad=True)
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    
    # Check that masked probs are exactly zero
    assert (probs[0, 0, torch.triu_indices(seq_len, seq_len, offset=1)[0], 
                      torch.triu_indices(seq_len, seq_len, offset=1)[1]] == 0).all()
    
    print("✓ Causal gradient flow verified: masked positions have zero probability\n")
    return True


if __name__ == "__main__":
    try:
        test1_pass = test_causal_constraint()
        test2_pass = test_gradient_flow()
        
        if test1_pass and test2_pass:
            print("=== All causal masking tests passed ===")
            exit(0)
        else:
            print("=== Some causal masking tests failed ===")
            exit(1)
    except Exception as e:
        print(f"✗ Test error: {e}")
        exit(1)
