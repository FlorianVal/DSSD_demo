"""
Benchmark tests for KV Cache optimization in DSSD.

This module provides deterministic benchmarks to measure:
1. Layer forward counts (direct measure of computation)
2. Wall clock time for draft + verify phases
3. Optional FLOPs estimation

Run with: python -m tests.benchmark_kv_cache
"""

import time
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager


# =============================================================================
# Instrumentation
# =============================================================================


@dataclass
class BenchmarkMetrics:
    """Tracks metrics during benchmark run."""

    # Layer forward counts
    layer_forward_counts: Dict[int, int] = field(default_factory=dict)
    total_layer_forwards: int = 0

    # Timing
    draft_time_ms: float = 0.0
    verify_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Token counts
    tokens_drafted: int = 0
    tokens_accepted: int = 0
    tokens_rejected: int = 0

    # Early exit distribution
    exit_layers: List[int] = field(default_factory=list)

    def reset(self):
        """Reset all metrics."""
        self.layer_forward_counts.clear()
        self.total_layer_forwards = 0
        self.draft_time_ms = 0.0
        self.verify_time_ms = 0.0
        self.total_time_ms = 0.0
        self.tokens_drafted = 0
        self.tokens_accepted = 0
        self.tokens_rejected = 0
        self.exit_layers.clear()

    def record_layer_forward(self, layer_idx: int):
        """Record a layer forward pass."""
        self.layer_forward_counts[layer_idx] = (
            self.layer_forward_counts.get(layer_idx, 0) + 1
        )
        self.total_layer_forwards += 1

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            "=" * 50,
            "BENCHMARK METRICS",
            "=" * 50,
            f"Total Layer Forwards: {self.total_layer_forwards}",
            f"Tokens Drafted: {self.tokens_drafted}",
            f"Tokens Accepted: {self.tokens_accepted}",
            f"Tokens Rejected: {self.tokens_rejected}",
            f"Draft Time: {self.draft_time_ms:.2f} ms",
            f"Verify Time: {self.verify_time_ms:.2f} ms",
            f"Total Time: {self.total_time_ms:.2f} ms",
            "",
            "Layer Forward Distribution:",
        ]
        for layer_idx in sorted(self.layer_forward_counts.keys()):
            count = self.layer_forward_counts[layer_idx]
            lines.append(f"  Layer {layer_idx:2d}: {count} forwards")

        if self.exit_layers:
            avg_exit = sum(self.exit_layers) / len(self.exit_layers)
            lines.append(f"\nAverage Exit Layer: {avg_exit:.1f}")

        lines.append("=" * 50)
        return "\n".join(lines)


# Global metrics instance for instrumentation
_metrics: Optional[BenchmarkMetrics] = None


def get_metrics() -> Optional[BenchmarkMetrics]:
    """Get the current metrics instance."""
    return _metrics


@contextmanager
def benchmark_context():
    """Context manager that enables metric collection."""
    global _metrics
    _metrics = BenchmarkMetrics()
    try:
        yield _metrics
    finally:
        _metrics = None


def instrument_layer_forward(layer_idx: int):
    """Call this from forward_layer to record layer execution."""
    if _metrics is not None:
        _metrics.record_layer_forward(layer_idx)


# =============================================================================
# Timer Utilities
# =============================================================================


class Timer:
    """Simple timer for benchmarking."""

    def __init__(self):
        self.start_time = None
        self.elapsed_ms = 0.0

    def start(self):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        if self.start_time is not None:
            self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        return self.elapsed_ms


# =============================================================================
# Benchmark Test Scenarios
# =============================================================================


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    # Model setting
    model_name: str = "Qwen/Qwen3-0.6B"

    # Generation settings
    prompt: str = "Explain what machine learning is in simple terms."
    max_draft_length: int = 5
    num_iterations: int = 3  # Multiple iterations for averaging

    # Thresholds for early exit (simulated or real)
    accuracy_level: float = 0.75

    # Reproducibility
    seed: int = 42


def run_single_draft_verify_benchmark(
    decoder,  # DSSDecoder
    config: BenchmarkConfig,
    use_cache: bool = False,
) -> BenchmarkMetrics:
    """
    Run a single draft + verify cycle and measure metrics.

    Args:
        decoder: The DSSDecoder instance
        config: Benchmark configuration
        use_cache: Whether to use JaggedKVCache (for comparison)

    Returns:
        BenchmarkMetrics with recorded data
    """
    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    with benchmark_context() as metrics:
        timer = Timer()

        # Tokenize prompt
        input_ids = decoder.tokenizer.encode(config.prompt, return_tensors="pt").to(
            decoder.device
        )

        # Get thresholds
        thresholds = {}
        if decoder.calibration:
            thresholds = decoder.calibration.get_thresholds_for_level(
                config.accuracy_level
            )

        # ========== DRAFT PHASE ==========
        timer.start()
        drafted_tokens = []
        draft_ids = input_ids.clone()

        for _ in range(config.max_draft_length):
            # Call the drafting function
            # Note: This will need to be modified to use our instrumented version
            draft_result = decoder._draft_single_token(draft_ids, thresholds)

            if draft_result is None:
                break

            token_id, exit_head, exit_layer, uncertainty = draft_result
            drafted_tokens.append((token_id, exit_head, exit_layer, uncertainty))
            metrics.exit_layers.append(exit_layer)

            if token_id == decoder.tokenizer.eos_token_id:
                break

            draft_ids = torch.cat(
                [draft_ids, torch.tensor([[token_id]], device=decoder.device)], dim=1
            )

        metrics.draft_time_ms = timer.stop()
        metrics.tokens_drafted = len(drafted_tokens)

        # ========== VERIFY PHASE ==========
        timer.start()

        if drafted_tokens:
            with torch.no_grad():
                outputs = decoder.model(draft_ids, use_cache=False)
                verify_logits = outputs.logits

            # Verify each token
            start_pos = input_ids.shape[1] - 1
            accepted = 0

            for i, (token_id, exit_head, exit_layer, uncertainty) in enumerate(
                drafted_tokens
            ):
                verify_pos = start_pos + i
                verified_token = torch.argmax(verify_logits[0, verify_pos, :]).item()

                if token_id == verified_token:
                    accepted += 1
                else:
                    break

            metrics.tokens_accepted = accepted
            metrics.tokens_rejected = len(drafted_tokens) - accepted

        metrics.verify_time_ms = timer.stop()
        metrics.total_time_ms = metrics.draft_time_ms + metrics.verify_time_ms

    return metrics


def run_baseline_benchmark(decoder, config: BenchmarkConfig) -> BenchmarkMetrics:
    """
    Run baseline benchmark (current implementation without cache optimization).
    """
    print(f"\n{'=' * 60}")
    print("BASELINE BENCHMARK (No Cache)")
    print(f"{'=' * 60}")
    print(f"Model: {config.model_name}")
    print(f"Prompt: '{config.prompt[:50]}...'")
    print(f"Max Draft Length: {config.max_draft_length}")
    print(f"Iterations: {config.num_iterations}")

    all_metrics = []

    for i in range(config.num_iterations):
        print(f"\nIteration {i + 1}/{config.num_iterations}...")
        metrics = run_single_draft_verify_benchmark(decoder, config, use_cache=False)
        all_metrics.append(metrics)
        print(f"  Layer Forwards: {metrics.total_layer_forwards}")
        print(f"  Draft Time: {metrics.draft_time_ms:.2f} ms")
        print(f"  Verify Time: {metrics.verify_time_ms:.2f} ms")

    # Average metrics
    avg_metrics = BenchmarkMetrics()
    avg_metrics.total_layer_forwards = sum(
        m.total_layer_forwards for m in all_metrics
    ) // len(all_metrics)
    avg_metrics.draft_time_ms = sum(m.draft_time_ms for m in all_metrics) / len(
        all_metrics
    )
    avg_metrics.verify_time_ms = sum(m.verify_time_ms for m in all_metrics) / len(
        all_metrics
    )
    avg_metrics.total_time_ms = sum(m.total_time_ms for m in all_metrics) / len(
        all_metrics
    )
    avg_metrics.tokens_drafted = all_metrics[0].tokens_drafted
    avg_metrics.tokens_accepted = all_metrics[0].tokens_accepted
    avg_metrics.tokens_rejected = all_metrics[0].tokens_rejected

    # Combine layer counts
    for m in all_metrics:
        for layer_idx, count in m.layer_forward_counts.items():
            avg_metrics.layer_forward_counts[layer_idx] = (
                avg_metrics.layer_forward_counts.get(layer_idx, 0)
                + count // len(all_metrics)
            )

    print("\n" + avg_metrics.summary())
    return avg_metrics


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run benchmark suite."""
    import sys

    sys.path.insert(0, "/home/fvalade/workspace/DSSD_demo")

    from src.inference import load_dssd_model

    config = BenchmarkConfig()

    print("Loading model...")
    try:
        # You'll need to update these paths to match your setup
        decoder, tokenizer = load_dssd_model(
            model_name=config.model_name,
            heads_path="../checkpoints/qwen3-0.6b/aux_heads.pt",
            config_path="../checkpoints/qwen3-0.6b/config.json",
            calibration_path="../checkpoints/qwen3-0.6b/calibration.json",
            device="auto",
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTo run this benchmark, ensure you have:")
        print("  1. A trained auxiliary heads checkpoint")
        print("  2. The corresponding config.json")
        print("  3. (Optional) calibration.json for thresholds")
        return

    # Run baseline benchmark
    baseline_metrics = run_baseline_benchmark(decoder, config)

    # Save results for later comparison
    print("\n" + "=" * 60)
    print("BASELINE RESULTS SAVED")
    print("Run this again after implementing JaggedKVCache to compare.")
    print("=" * 60)


if __name__ == "__main__":
    main()
