"""
Step-by-step verification tests for KV Cache operations.

These tests verify the correctness of the JaggedKVCache implementation
without requiring a full model. Run with: pytest tests/test_cache_operations.py -v
"""

import pytest
import torch
from typing import List, Tuple, Optional


# =============================================================================
# Mock Cache Implementation (to be replaced with real JaggedKVCache)
# =============================================================================


class JaggedKVCache:
    """
    Jagged KV Cache that tracks per-layer sequence lengths.

    This is a reference implementation for testing. The production version
    will be in src/jagged_cache.py.
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int = 1,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Per-layer storage: List of (key_cache, value_cache) or None
        self.layer_caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            None for _ in range(num_layers)
        ]

        # Track sequence length per layer
        self.layer_seq_lengths: List[int] = [0] * num_layers

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache for a layer at specific positions.

        Args:
            layer_idx: Layer index to update
            key_states: [B, num_kv_heads, seq_len, head_dim]
            value_states: [B, num_kv_heads, seq_len, head_dim]
            cache_position: [seq_len] positions to update

        Returns:
            (full_keys, full_values) including cached + new
        """
        new_len = cache_position[-1].item() + 1
        input_seq_len = key_states.shape[2]

        if self.layer_caches[layer_idx] is None:
            # First time - check if positions are contiguous starting from 0
            if cache_position[0].item() == 0 and input_seq_len == new_len:
                # Simple case: positions [0, 1, ..., n-1] - just clone
                self.layer_caches[layer_idx] = (
                    key_states.clone(),
                    value_states.clone(),
                )
            else:
                # Non-contiguous or not starting from 0 - need to allocate full size
                k_cache = torch.zeros(
                    (self.batch_size, self.num_kv_heads, new_len, self.head_dim),
                    device=self.device,
                    dtype=self.dtype,
                )
                v_cache = torch.zeros(
                    (self.batch_size, self.num_kv_heads, new_len, self.head_dim),
                    device=self.device,
                    dtype=self.dtype,
                )
                k_cache[:, :, cache_position.long(), :] = key_states
                v_cache[:, :, cache_position.long(), :] = value_states
                self.layer_caches[layer_idx] = (k_cache, v_cache)
            self.layer_seq_lengths[layer_idx] = new_len
        else:
            k_cache, v_cache = self.layer_caches[layer_idx]
            current_len = k_cache.shape[2]

            if new_len > current_len:
                # Need to extend cache
                extension_size = new_len - current_len
                k_extension = torch.zeros(
                    (self.batch_size, self.num_kv_heads, extension_size, self.head_dim),
                    device=self.device,
                    dtype=self.dtype,
                )
                v_extension = torch.zeros(
                    (self.batch_size, self.num_kv_heads, extension_size, self.head_dim),
                    device=self.device,
                    dtype=self.dtype,
                )
                k_cache = torch.cat([k_cache, k_extension], dim=2)
                v_cache = torch.cat([v_cache, v_extension], dim=2)

            # Update at cache_position (handles both extension and gap-filling)
            k_cache[:, :, cache_position.long(), :] = key_states
            v_cache[:, :, cache_position.long(), :] = value_states

            self.layer_caches[layer_idx] = (k_cache, v_cache)
            self.layer_seq_lengths[layer_idx] = max(
                self.layer_seq_lengths[layer_idx], new_len
            )

        return self.layer_caches[layer_idx]

    def get_kv(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get cached KV for a layer, or None if not cached."""
        return self.layer_caches[layer_idx]

    def get_seq_length(self, layer_idx: int) -> int:
        """Get the sequence length cached for a layer."""
        return self.layer_seq_lengths[layer_idx]

    def truncate_from(self, position: int):
        """
        Truncate all layer caches from position onwards.
        Used for rollback on rejection.
        """
        for layer_idx in range(self.num_layers):
            if self.layer_caches[layer_idx] is not None:
                k, v = self.layer_caches[layer_idx]
                if k.shape[2] > position:
                    self.layer_caches[layer_idx] = (
                        k[:, :, :position, :],
                        v[:, :, :position, :],
                    )
                    self.layer_seq_lengths[layer_idx] = min(
                        self.layer_seq_lengths[layer_idx], position
                    )

    def clone(self) -> "JaggedKVCache":
        """Create a deep copy of the cache for speculation."""
        new_cache = JaggedKVCache(
            num_layers=self.num_layers,
            batch_size=self.batch_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        for i, kv in enumerate(self.layer_caches):
            if kv is not None:
                new_cache.layer_caches[i] = (kv[0].clone(), kv[1].clone())
        new_cache.layer_seq_lengths = self.layer_seq_lengths.copy()
        return new_cache

    def get_missing_layers(self, position: int, target_layer: int) -> List[int]:
        """
        Get list of layers that need computation for this position.

        Args:
            position: The position we need KV for
            target_layer: The deepest layer we need to reach

        Returns:
            List of layer indices that need to be computed
        """
        missing = []
        for layer_idx in range(target_layer + 1):
            if self.layer_seq_lengths[layer_idx] <= position:
                missing.append(layer_idx)
        return missing

    def __repr__(self):
        lines = [f"JaggedKVCache(num_layers={self.num_layers})"]
        for i in range(self.num_layers):
            seq_len = self.layer_seq_lengths[i]
            lines.append(f"  Layer {i:2d}: {seq_len} positions cached")
        return "\n".join(lines)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def small_cache():
    """Create a small cache for testing."""
    return JaggedKVCache(
        num_layers=8,
        batch_size=1,
        num_kv_heads=4,
        head_dim=64,
        device="cpu",
        dtype=torch.float32,
    )


@pytest.fixture
def sample_kv():
    """Create sample KV tensors."""

    def _make_kv(batch_size=1, num_heads=4, seq_len=1, head_dim=64):
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        return k, v

    return _make_kv


# =============================================================================
# Test 1: Basic Cache Operations
# =============================================================================


class TestCacheBasicOperations:
    """Test basic cache update and retrieval."""

    def test_cache_starts_empty(self, small_cache):
        """Cache should start with no entries."""
        for i in range(small_cache.num_layers):
            assert small_cache.get_kv(i) is None
            assert small_cache.get_seq_length(i) == 0

    def test_single_position_update(self, small_cache, sample_kv):
        """Test updating cache with a single position."""
        k, v = sample_kv()
        cache_position = torch.tensor([0])

        small_cache.update(
            layer_idx=0, key_states=k, value_states=v, cache_position=cache_position
        )

        assert small_cache.get_kv(0) is not None
        assert small_cache.get_seq_length(0) == 1
        assert small_cache.get_kv(1) is None  # Other layers unchanged

    def test_multiple_positions_update(self, small_cache, sample_kv):
        """Test updating cache with multiple positions at once."""
        k, v = sample_kv(seq_len=3)
        cache_position = torch.tensor([0, 1, 2])

        small_cache.update(
            layer_idx=0, key_states=k, value_states=v, cache_position=cache_position
        )

        assert small_cache.get_seq_length(0) == 3
        cached_k, cached_v = small_cache.get_kv(0)
        assert cached_k.shape[2] == 3

    def test_extending_cache(self, small_cache, sample_kv):
        """Test extending cache with new positions."""
        # First update
        k1, v1 = sample_kv(seq_len=2)
        small_cache.update(0, k1, v1, torch.tensor([0, 1]))

        # Extend with more positions
        k2, v2 = sample_kv(seq_len=2)
        small_cache.update(0, k2, v2, torch.tensor([2, 3]))

        assert small_cache.get_seq_length(0) == 4
        cached_k, _ = small_cache.get_kv(0)
        assert cached_k.shape[2] == 4


# =============================================================================
# Test 2: Jagged Cache Behavior
# =============================================================================


class TestJaggedCacheBehavior:
    """Test that cache correctly handles different layers with different lengths."""

    def test_different_layers_different_lengths(self, small_cache, sample_kv):
        """Simulate early exit where different layers have different cached lengths.

        Note: seq_length tracks capacity (max_pos + 1), not filled count.
        When layer 3 is first updated at position [1], it allocates space for
        positions [0, 1], but position 0 contains zeros (unfilled).
        The lazy fill mechanism will fill these gaps when needed.
        """
        # Token 0: Exit at layer 2 -> layers 0-2 get cached
        for layer_idx in range(3):
            k, v = sample_kv()
            small_cache.update(layer_idx, k, v, torch.tensor([0]))

        # Token 1: Exit at layer 4 -> layers 0-4 get cached
        for layer_idx in range(5):
            k, v = sample_kv()
            small_cache.update(layer_idx, k, v, torch.tensor([1]))

        # Check jagged structure
        # seq_length = capacity = max_position + 1
        assert small_cache.get_seq_length(0) == 2  # Both tokens
        assert small_cache.get_seq_length(1) == 2
        assert small_cache.get_seq_length(2) == 2
        # Layers 3-4 have capacity 2 (allocated for positions 0,1)
        # Position 0 is zeros (unfilled) - will be lazy-filled when needed
        assert small_cache.get_seq_length(3) == 2
        assert small_cache.get_seq_length(4) == 2
        assert small_cache.get_seq_length(5) == 0  # Never reached

    def test_get_missing_layers(self, small_cache, sample_kv):
        """Test detecting which layers need computation."""
        # Cache position 0 for layers 0-2 only
        for layer_idx in range(3):
            k, v = sample_kv()
            small_cache.update(layer_idx, k, v, torch.tensor([0]))

        # Check what's missing for position 0 up to layer 5
        missing = small_cache.get_missing_layers(position=0, target_layer=5)
        assert missing == [3, 4, 5]  # Layers 3-5 are missing

        # Check for position 1 (not cached anywhere)
        missing = small_cache.get_missing_layers(position=1, target_layer=5)
        assert missing == [0, 1, 2, 3, 4, 5]  # All layers missing


# =============================================================================
# Test 3: Truncation for Rollback
# =============================================================================


class TestCacheTruncation:
    """Test cache truncation for rejection rollback."""

    def test_truncate_removes_positions(self, small_cache, sample_kv):
        """Test that truncation removes positions correctly."""
        # Fill cache with 5 positions
        for pos in range(5):
            k, v = sample_kv()
            small_cache.update(0, k, v, torch.tensor([pos]))

        assert small_cache.get_seq_length(0) == 5

        # Truncate at position 3 (keep 0, 1, 2)
        small_cache.truncate_from(3)

        assert small_cache.get_seq_length(0) == 3
        cached_k, _ = small_cache.get_kv(0)
        assert cached_k.shape[2] == 3

    def test_truncate_all_layers(self, small_cache, sample_kv):
        """Test that truncation affects all layers."""
        # Fill multiple layers with different lengths
        for layer_idx in range(3):
            for pos in range(5):
                k, v = sample_kv()
                small_cache.update(layer_idx, k, v, torch.tensor([pos]))

        # Add more to layer 0
        for pos in range(5, 8):
            k, v = sample_kv()
            small_cache.update(0, k, v, torch.tensor([pos]))

        assert small_cache.get_seq_length(0) == 8
        assert small_cache.get_seq_length(1) == 5
        assert small_cache.get_seq_length(2) == 5

        # Truncate at position 4
        small_cache.truncate_from(4)

        assert small_cache.get_seq_length(0) == 4
        assert small_cache.get_seq_length(1) == 4
        assert small_cache.get_seq_length(2) == 4


# =============================================================================
# Test 4: Clone for Speculation
# =============================================================================


class TestCacheCloning:
    """Test cache cloning for speculative drafting."""

    def test_clone_creates_independent_copy(self, small_cache, sample_kv):
        """Test that clone creates truly independent copy."""
        # Fill original cache
        k, v = sample_kv(seq_len=3)
        small_cache.update(0, k, v, torch.tensor([0, 1, 2]))

        # Clone
        cloned = small_cache.clone()

        # Modify original
        k2, v2 = sample_kv()
        small_cache.update(0, k2, v2, torch.tensor([3]))

        # Check clone is unchanged
        assert small_cache.get_seq_length(0) == 4
        assert cloned.get_seq_length(0) == 3

    def test_clone_preserves_data(self, small_cache, sample_kv):
        """Test that clone preserves actual tensor values."""
        k, v = sample_kv()
        small_cache.update(0, k, v, torch.tensor([0]))

        cloned = small_cache.clone()

        orig_k, orig_v = small_cache.get_kv(0)
        clone_k, clone_v = cloned.get_kv(0)

        assert torch.allclose(orig_k, clone_k)
        assert torch.allclose(orig_v, clone_v)


# =============================================================================
# Test 5: Simulated Draft/Verify Scenario
# =============================================================================


class TestDraftVerifyScenario:
    """Simulate a realistic draft/verify scenario."""

    def test_draft_verify_with_full_accept(self, small_cache, sample_kv):
        """Simulate drafting 3 tokens, all accepted."""
        # Prompt prefill (position 0-4)
        for pos in range(5):
            for layer_idx in range(small_cache.num_layers):
                k, v = sample_kv()
                small_cache.update(layer_idx, k, v, torch.tensor([pos]))

        # Clone for drafting
        draft_cache = small_cache.clone()

        # Draft 3 tokens (positions 5, 6, 7), exiting at different layers
        exit_layers = [2, 4, 3]  # Token 5 exits at layer 2, etc.

        for i, (pos, exit_layer) in enumerate(zip([5, 6, 7], exit_layers)):
            for layer_idx in range(exit_layer + 1):
                k, v = sample_kv()
                draft_cache.update(layer_idx, k, v, torch.tensor([pos]))

        # Check jagged structure after drafting
        assert draft_cache.get_seq_length(0) == 8  # All 8 positions
        assert draft_cache.get_seq_length(2) == 8  # All tokens reached layer 2
        assert draft_cache.get_seq_length(4) == 7  # Only tokens 5,6 reached layer 4

        # "Verification" - all accepted, fill remaining layers
        for pos in [5, 6, 7]:
            for layer_idx in range(small_cache.num_layers):
                if draft_cache.get_seq_length(layer_idx) <= pos:
                    k, v = sample_kv()
                    draft_cache.update(layer_idx, k, v, torch.tensor([pos]))

        # After verification, all layers should have all positions
        for layer_idx in range(small_cache.num_layers):
            assert draft_cache.get_seq_length(layer_idx) == 8

    def test_draft_verify_with_rejection(self, small_cache, sample_kv):
        """Simulate drafting 3 tokens, rejected at position 6."""
        # Prompt prefill
        for pos in range(5):
            for layer_idx in range(small_cache.num_layers):
                k, v = sample_kv()
                small_cache.update(layer_idx, k, v, torch.tensor([pos]))

        # Clone for drafting
        draft_cache = small_cache.clone()

        # Draft 3 tokens
        for pos in [5, 6, 7]:
            for layer_idx in range(3):  # All exit at layer 2
                k, v = sample_kv()
                draft_cache.update(layer_idx, k, v, torch.tensor([pos]))

        # Simulate rejection at position 6
        # Accept position 5, reject 6 (and 7)
        draft_cache.truncate_from(6)

        # Should only have positions 0-5
        assert draft_cache.get_seq_length(0) == 6
        assert draft_cache.get_seq_length(1) == 6
        assert draft_cache.get_seq_length(2) == 6


# =============================================================================
# Run tests directly
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
