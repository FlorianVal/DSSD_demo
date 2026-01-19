"""
Integration tests for JaggedKVCache with inference pipeline.

Run with: pytest tests/test_cache_integration.py -v
"""

import pytest
import torch
from typing import List, Optional

# Import from production module
import sys

sys.path.insert(0, "/home/fvalade/workspace/DSSD_demo")

from src.jagged_cache import JaggedKVCache


class TestJaggedKVCacheProduction:
    """Test the production JaggedKVCache implementation."""

    @pytest.fixture
    def cache(self):
        """Create a test cache."""
        return JaggedKVCache(
            num_layers=8,
            batch_size=1,
            num_kv_heads=4,
            head_dim=64,
            device="cpu",
            dtype=torch.float32,
        )

    @pytest.fixture
    def sample_kv(self):
        """Create sample KV tensors."""

        def _make_kv(batch_size=1, num_heads=4, seq_len=1, head_dim=64):
            k = torch.randn(batch_size, num_heads, seq_len, head_dim)
            v = torch.randn(batch_size, num_heads, seq_len, head_dim)
            return k, v

        return _make_kv

    def test_filled_positions_tracking(self, cache, sample_kv):
        """Test that filled_positions correctly tracks which positions are filled."""
        # Update layer 0 with position 0
        k, v = sample_kv()
        cache.update(0, k, v, torch.tensor([0]))

        assert cache.has_position(0, 0) == True
        assert cache.has_position(0, 1) == False
        assert cache.has_position(1, 0) == False  # Layer 1 not touched

    def test_needs_fill(self, cache, sample_kv):
        """Test needs_fill correctly identifies missing positions."""
        # Fill layer 0 with position 0
        k, v = sample_kv()
        cache.update(0, k, v, torch.tensor([0]))

        # Layer 0 has position 0, doesn't need fill
        assert cache.needs_fill(0, [0]) == False

        # Layer 0 doesn't have position 1
        assert cache.needs_fill(0, [1]) == True

        # Layer 1 has nothing
        assert cache.needs_fill(1, [0]) == True

    def test_get_unfilled_positions(self, cache, sample_kv):
        """Test getting unfilled positions."""
        # Fill positions 0 and 2 for layer 0
        k, v = sample_kv()
        cache.update(0, k, v, torch.tensor([0]))
        k, v = sample_kv()
        cache.update(0, k, v, torch.tensor([2]))

        # Unfilled up to position 4 should be [1, 3]
        unfilled = cache.get_unfilled_positions(0, 4)
        assert unfilled == [1, 3]

    def test_truncate_clears_filled_positions(self, cache, sample_kv):
        """Test that truncation also clears filled_positions."""
        # Fill positions 0-4
        for pos in range(5):
            k, v = sample_kv()
            cache.update(0, k, v, torch.tensor([pos]))

        assert cache.has_position(0, 4) == True

        # Truncate at position 3
        cache.truncate_from(3)

        # Positions 3 and 4 should be gone
        assert cache.has_position(0, 2) == True
        assert cache.has_position(0, 3) == False
        assert cache.has_position(0, 4) == False

    def test_clone_copies_filled_positions(self, cache, sample_kv):
        """Test that clone also copies filled_positions."""
        k, v = sample_kv()
        cache.update(0, k, v, torch.tensor([0]))

        cloned = cache.clone()

        assert cloned.has_position(0, 0) == True

        # Modify original
        k, v = sample_kv()
        cache.update(0, k, v, torch.tensor([1]))

        # Clone should be unaffected
        assert cache.has_position(0, 1) == True
        assert cloned.has_position(0, 1) == False

    def test_reset(self, cache, sample_kv):
        """Test that reset clears everything."""
        k, v = sample_kv()
        cache.update(0, k, v, torch.tensor([0]))

        cache.reset()

        assert cache.get_kv(0) is None
        assert cache.get_seq_length(0) == 0
        assert cache.has_position(0, 0) == False


class TestLazyFillScenario:
    """Test realistic lazy fill scenarios."""

    @pytest.fixture
    def cache(self):
        return JaggedKVCache(
            num_layers=8,
            batch_size=1,
            num_kv_heads=4,
            head_dim=64,
            device="cpu",
            dtype=torch.float32,
        )

    @pytest.fixture
    def sample_kv(self):
        def _make_kv(batch_size=1, num_heads=4, seq_len=1, head_dim=64):
            k = torch.randn(batch_size, num_heads, seq_len, head_dim)
            v = torch.randn(batch_size, num_heads, seq_len, head_dim)
            return k, v

        return _make_kv

    def test_lazy_fill_scenario(self, cache, sample_kv):
        """
        Simulate:
        - Prefill prompt (positions 0-4) through all layers
        - Draft token 5 exiting at layer 2
        - Draft token 6 exiting at layer 6 (needs lazy fill)
        """
        # Prefill: positions 0-4 through all 8 layers
        for pos in range(5):
            for layer_idx in range(8):
                k, v = sample_kv()
                cache.update(layer_idx, k, v, torch.tensor([pos]))

        # Verify prefill complete
        for layer_idx in range(8):
            assert cache.get_seq_length(layer_idx) == 5
            for pos in range(5):
                assert cache.has_position(layer_idx, pos)

        # Draft token 5, exit at layer 2
        for layer_idx in range(3):  # Layers 0, 1, 2
            k, v = sample_kv()
            cache.update(layer_idx, k, v, torch.tensor([5]))

        # Position 5 is filled only for layers 0-2
        assert cache.has_position(0, 5)
        assert cache.has_position(2, 5)
        assert not cache.has_position(3, 5)

        # Draft token 6, need to exit at layer 6
        # Check what positions are missing for layer 6
        missing_at_layer_6 = cache.get_missing_layers(5, 6)

        # Layers 3-6 are missing position 5
        assert 3 in missing_at_layer_6
        assert 6 in missing_at_layer_6
        assert 0 not in missing_at_layer_6  # Layer 0 has position 5

        # Check unfilled positions for layer 6 up to position 6
        unfilled = cache.get_unfilled_positions(6, 6)
        assert 5 in unfilled  # Position 5 is unfilled at layer 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
