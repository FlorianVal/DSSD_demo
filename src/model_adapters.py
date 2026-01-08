# Model Adapters for True Early Exit
# Abstract interface to stop layer computation early across architectures

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict, Callable
import torch
import torch.nn as nn
from torch import Tensor


class ModelAdapter(ABC):
    """Abstract interface for model internals to enable true early exit."""

    @abstractmethod
    def get_embed_tokens(self, input_ids: Tensor) -> Tensor:
        """Get token embeddings."""
        ...

    @abstractmethod
    def get_layers(self) -> nn.ModuleList:
        """Get list of decoder layers."""
        ...

    @abstractmethod
    def get_num_layers(self) -> int:
        """Get total number of layers."""
        ...

    @abstractmethod
    def forward_layer(
        self,
        layer: nn.Module,
        hidden_states: Tensor,
        position_ids: Tensor,
        attention_mask: Optional[Tensor],
        past_key_value: Optional[Tuple],
        position_embeddings: Optional[Tuple],
        use_cache: bool = True,
        cache_position: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tuple]]:
        """Forward through a single layer, returning hidden states and optional KV cache."""
        ...

    @abstractmethod
    def apply_final_norm(self, hidden_states: Tensor) -> Tensor:
        """Apply final normalization before lm_head."""
        ...

    @abstractmethod
    def get_lm_head_output(self, hidden_states: Tensor) -> Tensor:
        """Get logits from lm_head."""
        ...

    @abstractmethod
    def get_position_embeddings(
        self, hidden_states: Tensor, position_ids: Tensor
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """Get rotary position embeddings (cos, sin) if applicable."""
        ...


class LlamaStyleAdapter(ModelAdapter):
    """
    Adapter for Llama-style architectures.
    Works for: Llama, Llama2, Llama3, Qwen, Qwen2, Qwen3, Mistral, Gemma

    These models share the same internal structure:
    - model.model.embed_tokens
    - model.model.layers (ModuleList of decoder layers)
    - model.model.norm (final RMSNorm)
    - model.lm_head
    - model.model.rotary_emb (RoPE embeddings)
    """

    def __init__(self, model):
        self.model = model
        self._base = model.model
        self._layers = self._base.layers
        self._embed = self._base.embed_tokens
        self._norm = self._base.norm
        self._lm_head = model.lm_head
        self._rotary = getattr(self._base, "rotary_emb", None)
        self._num_layers = len(self._layers)

    def get_embed_tokens(self, input_ids: Tensor) -> Tensor:
        return self._embed(input_ids)

    def get_layers(self) -> nn.ModuleList:
        return self._layers

    def get_num_layers(self) -> int:
        return self._num_layers

    def forward_layer(
        self,
        layer: nn.Module,
        hidden_states: Tensor,
        position_ids: Tensor,
        attention_mask: Optional[Tensor],
        past_key_value: Optional[Tuple],
        position_embeddings: Optional[Tuple],
        use_cache: bool = True,
        cache_position: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tuple]]:
        """Forward through a decoder layer."""
        layer_outputs = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
        )
        hidden_states = layer_outputs[0]
        new_kv = layer_outputs[1] if len(layer_outputs) > 1 else None
        return hidden_states, new_kv

    def apply_final_norm(self, hidden_states: Tensor) -> Tensor:
        return self._norm(hidden_states)

    def get_lm_head_output(self, hidden_states: Tensor) -> Tensor:
        return self._lm_head(hidden_states)

    def get_position_embeddings(
        self, hidden_states: Tensor, position_ids: Tensor
    ) -> Optional[Tuple[Tensor, Tensor]]:
        if self._rotary is not None:
            cos, sin = self._rotary(hidden_states, position_ids)
            return (cos, sin)
        return None


def get_adapter(model) -> ModelAdapter:
    """
    Factory function to get the appropriate adapter for a model.

    Currently supports Llama-style models (Llama, Qwen, Mistral, Gemma).
    """
    # Check for Llama-style architecture
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return LlamaStyleAdapter(model)

    # GPT-2 style (transformer.h)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        raise NotImplementedError("GPT-2 style models not yet supported")

    raise ValueError(f"Unsupported model architecture: {type(model)}")
