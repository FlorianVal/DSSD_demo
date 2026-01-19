"""
JaggedKVCache - Sparse KV Cache for Early Exit Inference.

This cache tracks per-layer sequence lengths, enabling efficient
generation with early exit heads that stop at different layers.
"""

import torch
from typing import List, Tuple, Optional


class JaggedKVCache:
    """
    Sparse KV Cache that tracks per-layer sequence lengths.

    Unlike standard KV caches where all layers have the same length,
    this cache allows different layers to have different cached lengths.
    This is essential for early exit inference where tokens may exit
    at different layers.

    Key features:
    - Per-layer KV storage with independent lengths
    - Lazy fill: missing positions are detected and can be computed on-demand
    - Truncation: efficient rollback on rejection
    - Cloning: snapshot for speculative drafting

    Attributes:
        num_layers: Total number of transformer layers
        batch_size: Batch size (typically 1 for inference)
        num_kv_heads: Number of key-value heads
        head_dim: Dimension of each head
        device: Device to store tensors on
        dtype: Data type for tensors
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

        # Track sequence length per layer (capacity = max_position + 1)
        self.layer_seq_lengths: List[int] = [0] * num_layers

        # Track which positions are actually filled (for lazy fill detection)
        # This is a list of sets, one per layer
        self.filled_positions: List[set] = [set() for _ in range(num_layers)]

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
            key_states: [B, num_kv_heads, seq_len, head_dim] new key states
            value_states: [B, num_kv_heads, seq_len, head_dim] new value states
            cache_position: [seq_len] tensor of positions to update

        Returns:
            (full_keys, full_values) tuple with all cached data
        """
        new_len = cache_position[-1].item() + 1
        input_seq_len = key_states.shape[2]
        positions = cache_position.tolist()

        if self.layer_caches[layer_idx] is None:
            # First time - check if positions are contiguous starting from 0
            if cache_position[0].item() == 0 and input_seq_len == new_len:
                # Simple case: positions [0, 1, ..., n-1] - just clone
                self.layer_caches[layer_idx] = (
                    key_states.clone(),
                    value_states.clone(),
                )
            else:
                # Non-contiguous or not starting from 0 - allocate full size
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

            # Update at cache_position
            k_cache[:, :, cache_position.long(), :] = key_states
            v_cache[:, :, cache_position.long(), :] = value_states

            self.layer_caches[layer_idx] = (k_cache, v_cache)
            self.layer_seq_lengths[layer_idx] = max(
                self.layer_seq_lengths[layer_idx], new_len
            )

        # Track filled positions
        self.filled_positions[layer_idx].update(positions)

        return self.layer_caches[layer_idx]

    def get_kv(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get cached KV for a layer, or None if not cached."""
        return self.layer_caches[layer_idx]

    def get_seq_length(self, layer_idx: int) -> int:
        """Get the sequence length (capacity) for a layer."""
        return self.layer_seq_lengths[layer_idx]

    def has_position(self, layer_idx: int, position: int) -> bool:
        """Check if a specific position is filled for a layer."""
        return position in self.filled_positions[layer_idx]

    def get_unfilled_positions(self, layer_idx: int, up_to: int) -> List[int]:
        """Get list of positions that are not filled for a layer, up to `up_to` (exclusive)."""
        all_positions = set(range(up_to))
        filled = self.filled_positions[layer_idx]
        return sorted(all_positions - filled)

    def needs_fill(self, layer_idx: int, positions: List[int]) -> bool:
        """Check if any of the given positions need to be filled for a layer."""
        return not all(p in self.filled_positions[layer_idx] for p in positions)

    def get_missing_layers(self, position: int, target_layer: int) -> List[int]:
        """
        Get list of layers that need computation for a position.

        Args:
            position: The position we need KV for
            target_layer: The deepest layer we need to reach

        Returns:
            List of layer indices that need computation for this position
        """
        missing = []
        for layer_idx in range(target_layer + 1):
            if position not in self.filled_positions[layer_idx]:
                missing.append(layer_idx)
        return missing

    def truncate_from(self, position: int):
        """
        Truncate all layer caches from position onwards (exclusive).
        Used for rollback on rejection.

        Args:
            position: First position to remove (keeps 0..position-1)
        """
        for layer_idx in range(self.num_layers):
            if self.layer_caches[layer_idx] is not None:
                k, v = self.layer_caches[layer_idx]
                if k.shape[2] > position:
                    self.layer_caches[layer_idx] = (
                        k[:, :, :position, :].contiguous(),
                        v[:, :, :position, :].contiguous(),
                    )
                    self.layer_seq_lengths[layer_idx] = min(
                        self.layer_seq_lengths[layer_idx], position
                    )

            # Remove filled positions >= position
            self.filled_positions[layer_idx] = {
                p for p in self.filled_positions[layer_idx] if p < position
            }

    def clone(self) -> "JaggedKVCache":
        """
        Create a deep copy of the cache for speculative drafting.

        Returns:
            Independent copy that can be modified without affecting original
        """
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
        new_cache.filled_positions = [s.copy() for s in self.filled_positions]
        return new_cache

    def reset(self):
        """Reset the cache to empty state."""
        self.layer_caches = [None for _ in range(self.num_layers)]
        self.layer_seq_lengths = [0] * self.num_layers
        self.filled_positions = [set() for _ in range(self.num_layers)]

    def __repr__(self) -> str:
        lines = [f"JaggedKVCache(num_layers={self.num_layers}, device={self.device})"]
        for i in range(min(self.num_layers, 10)):  # Show first 10 layers
            seq_len = self.layer_seq_lengths[i]
            filled = len(self.filled_positions[i])
            if seq_len > 0:
                lines.append(f"  Layer {i:2d}: {filled}/{seq_len} filled")
        if self.num_layers > 10:
            lines.append(f"  ... ({self.num_layers - 10} more layers)")
        return "\n".join(lines)
