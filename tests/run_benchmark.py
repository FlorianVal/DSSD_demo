#!/usr/bin/env python3
"""
Benchmark comparison: Standard generation vs Cache-optimized generation.

This script measures and compares:
- Layer forward counts
- Wall clock time
- Tokens per second

Usage:
    python tests/run_benchmark.py --model Qwen/Qwen3-0.6B --heads-path /path/to/heads.pt
"""

import argparse
import time
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def make_dummy_decoder():
    """Create a minimal decoder for benchmarking without GPU."""
    from src.jagged_cache import JaggedKVCache

    print("\n" + "=" * 60)
    print("BENCHMARK: JaggedKVCache Operations (No GPU Required)")
    print("=" * 60)

    # Test cache performance
    num_layers = 28
    batch_size = 1
    num_heads = 8
    head_dim = 128
    seq_len = 100

    cache = JaggedKVCache(
        num_layers=num_layers,
        batch_size=batch_size,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        device="cpu",
        dtype=torch.float32,
    )

    # Simulate prefill
    print(f"\nSimulating prefill ({seq_len} tokens, {num_layers} layers)...")
    start = time.perf_counter()
    for pos in range(seq_len):
        for layer_idx in range(num_layers):
            k = torch.randn(batch_size, num_heads, 1, head_dim)
            v = torch.randn(batch_size, num_heads, 1, head_dim)
            cache.update(layer_idx, k, v, torch.tensor([pos]))
    prefill_time = (time.perf_counter() - start) * 1000
    print(f"  Prefill time: {prefill_time:.2f} ms")

    # Simulate draft phase (early exit at different layers)
    print("\nSimulating draft phase (5 tokens, variable exit layers)...")
    exit_layers = [4, 8, 6, 12, 10]  # Simulate different exit layers
    draft_cache = cache.clone()

    start = time.perf_counter()
    for i, exit_layer in enumerate(exit_layers):
        pos = seq_len + i
        for layer_idx in range(exit_layer + 1):
            k = torch.randn(batch_size, num_heads, 1, head_dim)
            v = torch.randn(batch_size, num_heads, 1, head_dim)
            draft_cache.update(layer_idx, k, v, torch.tensor([pos]))
    draft_time = (time.perf_counter() - start) * 1000
    print(f"  Draft time: {draft_time:.2f} ms")

    # Print cache state
    print("\nCache state after drafting:")
    for layer_idx in [0, 4, 8, 12, 16, 20, 24, 27]:
        filled = len(draft_cache.filled_positions[layer_idx])
        print(f"  Layer {layer_idx:2d}: {filled} positions filled")

    # Simulate verification (fill all layers for all positions)
    print("\nSimulating verification (lazy fill + full model)...")
    start = time.perf_counter()
    for pos in range(seq_len, seq_len + 5):
        # Find missing layers
        missing = draft_cache.get_missing_layers(pos, num_layers - 1)
        for layer_idx in missing:
            k = torch.randn(batch_size, num_heads, 1, head_dim)
            v = torch.randn(batch_size, num_heads, 1, head_dim)
            draft_cache.update(layer_idx, k, v, torch.tensor([pos]))
    verify_time = (time.perf_counter() - start) * 1000
    print(f"  Verify time: {verify_time:.2f} ms")

    # Calculate and explain savings
    print("\n" + "=" * 60)
    print("ANALYSIS: Layer Operations")
    print("=" * 60)

    # Prefill ops (same for all approaches - one-time cost)
    prefill_ops = seq_len * num_layers
    print(f"\nPREFILL (one-time): {prefill_ops} layer ops")

    # Draft phase with early exit
    draft_ops = sum(exit_layer + 1 for exit_layer in exit_layers)
    draft_ops_full = 5 * num_layers  # Without early exit
    print(f"\nDRAFT PHASE (5 tokens):")
    print(f"  With early exit: {draft_ops} ops (avg {draft_ops / 5:.1f} layers/token)")
    print(f"  Without early exit: {draft_ops_full} ops ({num_layers} layers/token)")
    print(
        f"  Draft savings: {draft_ops_full - draft_ops} ops ({100 * (1 - draft_ops / draft_ops_full):.0f}% reduction)"
    )

    # The KEY benefit: with cache, each draft token is O(1 token * exit_layer)
    # Without cache, it would be O(seq_len * exit_layer) per token
    print(f"\nCACHE BENEFIT:")
    print(f"  Without cache, each draft would recompute {seq_len}-token context")
    print(f"  With cache, each draft processes only 1 new token")
    per_token_savings = seq_len - 1  # Positions we don't recompute
    total_context_savings = per_token_savings * draft_ops
    print(f"  Context reuse savings: ~{total_context_savings} avoided operations")

    # Verify phase
    verify_ops = 5 * num_layers
    print(f"\nVERIFY PHASE: {verify_ops} ops (fills all layers for drafted tokens)")

    print(f"\nTotal time: {prefill_time + draft_time + verify_time:.2f} ms")

    return True


def run_full_benchmark(model_name, heads_path, config_path, calibration_path=None):
    """Run full benchmark with actual model."""
    from src.inference import load_dssd_model

    print("\n" + "=" * 60)
    print(f"BENCHMARK: Full Model Comparison")
    print(f"Model: {model_name}")
    print("=" * 60)

    try:
        decoder, tokenizer = load_dssd_model(
            model_name=model_name,
            heads_path=heads_path,
            config_path=config_path,
            calibration_path=calibration_path,
            device="auto",
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

    prompt = "Explain what machine learning is in three sentences."
    max_tokens = 50

    # Warmup
    print("\nWarming up...")
    _ = decoder.generate(
        prompt, max_tokens=10, use_early_exit=False, use_chat_template=True
    )

    # Benchmark standard generation
    print("\nRunning standard generation (no cache)...")
    start = time.perf_counter()
    result_standard = decoder.generate(
        prompt,
        max_tokens=max_tokens,
        use_early_exit=True,
        accuracy_level=0.75,
        use_chat_template=True,
    )
    time_standard = time.perf_counter() - start

    # Benchmark cache-optimized generation (fast version)
    print("Running cache-optimized generation (fast)...")
    start = time.perf_counter()
    result_cached = decoder.generate_fast(
        prompt,
        max_tokens=max_tokens,
        accuracy_level=0.75,
        use_chat_template=True,
    )
    time_cached = time.perf_counter() - start

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\nStandard Generation:")
    print(f"  Tokens generated: {len(result_standard.tokens)}")
    print(f"  Time: {time_standard:.2f}s")
    print(f"  Tokens/sec: {len(result_standard.tokens) / time_standard:.2f}")
    print(f"  Avg exit layer: {result_standard.avg_exit_layer:.1f}")

    print("\nCache-Optimized Generation:")
    print(f"  Tokens generated: {len(result_cached.tokens)}")
    print(f"  Time: {time_cached:.2f}s")
    print(f"  Tokens/sec: {len(result_cached.tokens) / time_cached:.2f}")
    print(f"  Avg exit layer: {result_cached.avg_exit_layer:.1f}")
    if "total_drafted" in result_cached.exit_distribution:
        print(f"  Drafted: {result_cached.exit_distribution['total_drafted']}")
        print(f"  Accepted: {result_cached.exit_distribution['total_accepted']}")
        print(
            f"  Acceptance rate: {result_cached.exit_distribution['acceptance_rate']:.1%}"
        )

    print("\nSpeedup:")
    speedup = time_standard / time_cached if time_cached > 0 else 0
    print(f"  {speedup:.2f}x faster with cache")

    return True


def main():
    parser = argparse.ArgumentParser(description="Benchmark DSSD generation")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model name")
    parser.add_argument("--heads-path", help="Path to aux heads checkpoint")
    parser.add_argument("--config-path", help="Path to model config")
    parser.add_argument("--calibration-path", help="Path to calibration file")
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run CPU-only cache benchmark"
    )
    args = parser.parse_args()

    if args.cpu_only or not args.heads_path:
        # Run CPU-only cache operations benchmark
        make_dummy_decoder()
    else:
        # Run full benchmark with model
        run_full_benchmark(
            args.model,
            args.heads_path,
            args.config_path,
            args.calibration_path,
        )


if __name__ == "__main__":
    main()
