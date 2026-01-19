"""
Tests for the correct early exit inference loop behavior.

The inference loop should work as follows:

1. SINGLE FORWARD PASS per token attempt:
   - Process layers sequentially
   - At each head checkpoint, check if confident enough
   - If confident: EARLY EXIT - return token immediately (save compute)
   - If no head confident: continue to lm_head, return token from there
   - NEVER return None - always produce exactly one token per forward pass

2. SPECULATIVE DECODING:
   - Drafted tokens (from early exit heads) are unverified
   - When we eventually run to lm_head (full model), we verify all pending drafts
   - The lm_head pass also produces a BONUS token (the next prediction)
   - On mismatch: use full model's token, discard remaining drafts

Key invariants:
- _draft_single_token NEVER returns None
- When all drafts are accepted, we get N+1 tokens (N verified + 1 bonus)
- No redundant computation (never run layers twice for same token)
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
from typing import List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import DSSDecoder, TokenInfo, AuxiliaryHead, compute_entropy
from src.model_adapters import ModelAdapter
from src.model_config import ModelConfig, CalibrationResult


class MockAdapter(ModelAdapter):
    """Mock adapter for testing without a real model."""

    def __init__(self, num_layers: int = 8, hidden_size: int = 64, vocab_size: int = 100):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self._layers = nn.ModuleList([nn.Identity() for _ in range(num_layers)])
        self._embed = nn.Embedding(vocab_size, hidden_size)
        self._norm = nn.LayerNorm(hidden_size)
        self._lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Create a mapping from layer to index
        self._layer_to_idx = {layer: idx for idx, layer in enumerate(self._layers)}
        
        # Track calls for verification
        self.layer_calls = []
        self.final_norm_calls = 0
        self.lm_head_calls = 0

    def get_embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._embed(input_ids)

    def get_layers(self) -> nn.ModuleList:
        return self._layers

    def get_num_layers(self) -> int:
        return self.num_layers

    def forward_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Tuple],
        position_embeddings: Optional[Tuple],
        use_cache: bool = True,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        layer_idx = self._layer_to_idx.get(layer, -1)
        self.layer_calls.append(layer_idx)
        return hidden_states, None

    def apply_final_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.final_norm_calls += 1
        return self._norm(hidden_states)

    def get_lm_head_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.lm_head_calls += 1
        return self._lm_head(hidden_states)

    def get_position_embeddings(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        # Return dummy cos/sin embeddings
        seq_len = hidden_states.shape[1]
        cos = torch.ones(1, seq_len, self.hidden_size)
        sin = torch.zeros(1, seq_len, self.hidden_size)
        return (cos, sin)

    def reset_tracking(self):
        self.layer_calls = []
        self.final_norm_calls = 0
        self.lm_head_calls = 0


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        self.pad_token = "<pad>"
        self.chat_template = None  # Disable chat template
    
    def encode(self, text: str, return_tensors: str = None) -> torch.Tensor:
        # Simple mock encoding
        tokens = [ord(c) % self.vocab_size for c in text[:10]]
        if return_tensors == "pt":
            return torch.tensor([tokens])
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(chr(t + 65) for t in token_ids)


@pytest.fixture
def mock_model_config():
    """Create a mock model config with 2 heads."""
    return ModelConfig(
        model_name="mock-model",
        num_heads=2,
        head_layer_indices=[2, 5],  # Heads at layers 2 and 5
        quantization="none",
        hidden_size=64,
        vocab_size=100,
        num_hidden_layers=8,
    )


@pytest.fixture
def mock_calibration():
    """Create mock calibration with thresholds."""
    return CalibrationResult(
        model_config_path="mock",
        calibration_dataset="mock",
        calibration_samples=100,
        uncertainty_metric="entropy",
        accuracy_levels=[0.75],
        thresholds={
            "0.75": {
                "0": 0.5,  # Head 0 threshold
                "1": 0.7,  # Head 1 threshold
            }
        },
    )


@pytest.fixture
def mock_aux_heads():
    """Create mock auxiliary heads."""
    heads = nn.ModuleList([
        AuxiliaryHead(hidden_size=64, vocab_size=100),
        AuxiliaryHead(hidden_size=64, vocab_size=100),
    ])
    return heads


class MockModel:
    """Mock model that can be configured to return specific outputs."""
    
    def __init__(self):
        self._forward_fn = None
    
    def parameters(self):
        return iter([torch.zeros(1)])
    
    def set_forward(self, fn):
        """Set the forward function to use."""
        self._forward_fn = fn
    
    def __call__(self, input_ids, **kwargs):
        if self._forward_fn is not None:
            return self._forward_fn(input_ids, **kwargs)
        # Default: return zeros
        seq_len = input_ids.shape[1]
        class Output:
            def __init__(self):
                self.logits = torch.zeros(1, seq_len, 100)
        return Output()


class MockOutput:
    """Simple output wrapper."""
    def __init__(self, logits):
        self.logits = logits


@pytest.fixture
def mock_decoder(mock_model_config, mock_calibration, mock_aux_heads):
    """Create a decoder with mocked components."""
    adapter = MockAdapter(num_layers=8, hidden_size=64, vocab_size=100)
    tokenizer = MockTokenizer(vocab_size=100)
    
    # Create a configurable mock model
    mock_model = MockModel()
    
    decoder = DSSDecoder(
        model=mock_model,
        adapter=adapter,
        aux_heads=mock_aux_heads,
        tokenizer=tokenizer,
        model_config=mock_model_config,
        calibration=mock_calibration,
        device="cpu",
    )
    return decoder


class TestDraftSingleTokenNeverReturnsNone:
    """
    _draft_single_token should NEVER return None.
    
    It should always return a token:
    - From an early exit head if confident, OR
    - From the lm_head if no head is confident
    """

    def test_returns_token_when_head_confident(self, mock_decoder):
        """When a head is confident, return token with that head's info."""
        # Make head 0 very confident (low entropy)
        with patch.object(mock_decoder.aux_heads[0], 'forward') as mock_head:
            # Create logits with very peaked distribution (low entropy)
            logits = torch.zeros(1, 1, 100)
            logits[0, 0, 42] = 100.0  # Very confident about token 42
            mock_head.return_value = logits
            
            input_ids = torch.tensor([[1, 2, 3]])
            thresholds = {0: 0.5, 1: 0.7}
            
            result = mock_decoder._draft_single_token(input_ids, thresholds)
            
            assert result is not None, "_draft_single_token returned None!"
            token_id, exit_head, exit_layer, uncertainty = result
            assert token_id == 42
            assert exit_head == 0
            assert exit_layer == 2  # Head 0 is at layer 2

    def test_returns_token_from_lm_head_when_no_head_confident(self, mock_decoder):
        """
        When NO head is confident, should continue to lm_head and return token.
        This is the critical fix - currently the code returns None here.
        """
        # Make all heads NOT confident (high entropy)
        def make_uncertain_logits(*args, **kwargs):
            logits = torch.randn(1, 1, 100)  # Random = high entropy
            return logits
        
        for head in mock_decoder.aux_heads:
            head.forward = make_uncertain_logits
        
        input_ids = torch.tensor([[1, 2, 3]])
        thresholds = {0: 0.001, 1: 0.001}  # Very strict thresholds
        
        result = mock_decoder._draft_single_token(input_ids, thresholds)
        
        # THIS IS THE KEY ASSERTION - currently fails!
        assert result is not None, (
            "_draft_single_token returned None when no head was confident. "
            "It should have continued to lm_head and returned a token."
        )
        
        token_id, exit_head, exit_layer, uncertainty = result
        assert exit_head is None, "Token should be from lm_head, not a head"
        assert exit_layer == mock_decoder.adapter.get_num_layers()

    def test_no_redundant_computation_when_lm_head_used(self, mock_decoder):
        """
        When falling back to lm_head, layers should only be computed ONCE.
        The current bug: layers are computed in _draft_single_token,
        then computed AGAIN in the fallback full model call.
        """
        adapter = mock_decoder.adapter
        adapter.reset_tracking()
        
        # Make all heads NOT confident
        def make_uncertain_logits(*args, **kwargs):
            return torch.randn(1, 1, 100)
        
        for head in mock_decoder.aux_heads:
            head.forward = make_uncertain_logits
        
        input_ids = torch.tensor([[1, 2, 3]])
        thresholds = {0: 0.001, 1: 0.001}
        
        result = mock_decoder._draft_single_token(input_ids, thresholds)
        
        # Count how many times each layer was called
        layer_call_counts = {}
        for layer_idx in adapter.layer_calls:
            layer_call_counts[layer_idx] = layer_call_counts.get(layer_idx, 0) + 1
        
        # Each layer should be called exactly ONCE
        for layer_idx in range(adapter.num_layers):
            count = layer_call_counts.get(layer_idx, 0)
            assert count == 1, (
                f"Layer {layer_idx} was called {count} times. "
                "Should be exactly 1 (no redundant computation)."
            )


class TestBonusTokenOnFullVerification:
    """
    When we run to lm_head (for verification or no confident head),
    we should get N+1 tokens: N verified drafts + 1 bonus.
    """

    def test_bonus_token_when_all_drafts_accepted(self, mock_decoder):
        """
        If all drafted tokens are verified correct, we should get:
        - All drafted tokens (verified)
        - PLUS one bonus token from the last lm_head position
        """
        num_layers = mock_decoder.adapter.get_num_layers()
        
        # Scenario: 3 tokens drafted with early exit, then one from lm_head (triggers verify)
        # The lm_head token triggers verification of all previous drafts
        drafted_sequence = [
            (10, 0, 2, 0.1),  # token 10, head 0, layer 2 (early exit)
            (20, 1, 5, 0.2),  # token 20, head 1, layer 5 (early exit)
            (30, 1, 5, 0.3),  # token 30, head 1, layer 5 (early exit)
            (40, None, num_layers, 0.0),  # token 40, lm_head (triggers verify)
        ]
        
        draft_call_count = [0]
        
        def mock_draft(*args, **kwargs):
            if draft_call_count[0] < len(drafted_sequence):
                result = drafted_sequence[draft_call_count[0]]
                draft_call_count[0] += 1
                return result
            # Return EOS to stop
            return (mock_decoder.tokenizer.eos_token_id, None, num_layers, 0.0)
        
        # Mock the full model verification
        def mock_model_forward(input_ids, **kwargs):
            seq_len = input_ids.shape[1]
            logits = torch.zeros(1, seq_len, 100)
            
            # Make all drafted tokens verify correctly
            # base_pos = prompt length - 1 = 3 - 1 = 2
            base_pos = 2
            for i, (token_id, _, _, _) in enumerate(drafted_sequence):
                if i < len(drafted_sequence):
                    logits[0, base_pos + i, token_id] = 100.0
            
            # Bonus token prediction at last position
            logits[0, -1, 99] = 100.0  # Predict token 99 as bonus
            
            return MockOutput(logits)
        
        mock_decoder.model.set_forward(mock_model_forward)
        
        with patch.object(mock_decoder, '_draft_single_token', side_effect=mock_draft):
            input_ids = torch.tensor([[1, 2, 3]])
            thresholds = {0: 0.5, 1: 0.7}
            
            tokens = mock_decoder._generate_with_early_exit(
                input_ids, max_tokens=10, thresholds=thresholds
            )
        
        # Should get 5 tokens: 4 drafted/lm_head + 1 bonus
        assert len(tokens) >= 5, (
            f"Expected at least 5 tokens (4 drafted + 1 bonus), got {len(tokens)}. "
            f"Tokens: {[(t.token_id, t.exit_head) for t in tokens]}"
        )
        
        # First 3 should be early exit tokens
        assert tokens[0].token_id == 10
        assert tokens[0].exit_head == 0
        assert tokens[1].token_id == 20
        assert tokens[1].exit_head == 1
        assert tokens[2].token_id == 30
        assert tokens[2].exit_head == 1
        
        # 4th is the lm_head token that triggered verification
        assert tokens[3].token_id == 40
        assert tokens[3].exit_head is None
        
        # 5th is the bonus token
        assert tokens[4].token_id == 99, (
            f"5th token should be bonus token 99, got {tokens[4].token_id}"
        )
        assert tokens[4].exit_head is None


class TestVerificationOnMismatch:
    """Test that verification correctly handles mismatches."""

    def test_rejected_draft_uses_full_model_token(self, mock_decoder):
        """
        When a draft is rejected (mismatch), we should:
        1. Use the full model's token instead
        2. Discard remaining drafted tokens
        """
        num_layers = mock_decoder.adapter.get_num_layers()
        
        # Scenario: 3 early exit tokens drafted, then lm_head token triggers verify
        # The second drafted token will NOT match
        drafted_sequence = [
            (10, 0, 2, 0.1),  # Matches
            (20, 1, 5, 0.2),  # Will NOT match - full model says 25
            (30, 1, 5, 0.3),  # Should be discarded
            (40, None, num_layers, 0.0),  # lm_head triggers verification
        ]
        
        draft_call_count = [0]
        def mock_draft(*args, **kwargs):
            if draft_call_count[0] < len(drafted_sequence):
                result = drafted_sequence[draft_call_count[0]]
                draft_call_count[0] += 1
                return result
            # Return EOS to stop
            return (mock_decoder.tokenizer.eos_token_id, None, num_layers, 0.0)
        
        def mock_model_forward(input_ids, **kwargs):
            seq_len = input_ids.shape[1]
            logits = torch.zeros(1, seq_len, 100)
            
            # base_pos = prompt_len - 1 = 3 - 1 = 2
            base_pos = 2
            
            # First draft matches
            logits[0, base_pos, 10] = 100.0
            
            # Second draft does NOT match - full model says 25
            logits[0, base_pos + 1, 25] = 100.0  # Different from drafted 20!
            
            return MockOutput(logits)
        
        mock_decoder.model.set_forward(mock_model_forward)
        
        with patch.object(mock_decoder, '_draft_single_token', side_effect=mock_draft):
            input_ids = torch.tensor([[1, 2, 3]])
            thresholds = {0: 0.5, 1: 0.7}
            
            tokens = mock_decoder._generate_with_early_exit(
                input_ids, max_tokens=10, thresholds=thresholds
            )
        
        # Should get exactly 2 tokens: first accepted, second corrected
        # Third drafted token should be discarded
        assert len(tokens) >= 2, f"Expected at least 2 tokens, got {len(tokens)}"
        
        # First token: accepted draft
        assert tokens[0].token_id == 10
        assert tokens[0].exit_head == 0
        
        # Second token: full model's correction
        assert tokens[1].token_id == 25, (
            f"Second token should be full model's 25, not drafted 20. Got {tokens[1].token_id}"
        )
        assert tokens[1].exit_head is None, "Corrected token should have exit_head=None"


class TestEarlyExitSavesCompute:
    """Test that early exit actually skips layer computation."""

    def test_early_exit_stops_at_confident_layer(self, mock_decoder):
        """When head 0 (layer 2) is confident, layers 3-7 should NOT be computed."""
        adapter = mock_decoder.adapter
        adapter.reset_tracking()
        
        # Make head 0 (at layer 2) very confident
        with patch.object(mock_decoder.aux_heads[0], 'forward') as mock_head:
            logits = torch.zeros(1, 1, 100)
            logits[0, 0, 42] = 100.0
            mock_head.return_value = logits
            
            input_ids = torch.tensor([[1, 2, 3]])
            thresholds = {0: 10.0, 1: 10.0}  # High thresholds, easy to beat
            
            result = mock_decoder._draft_single_token(input_ids, thresholds)
        
        # Should have exited at layer 2
        assert result is not None
        _, exit_head, exit_layer, _ = result
        assert exit_layer == 2
        
        # Only layers 0, 1, 2 should have been called
        max_layer_called = max(adapter.layer_calls) if adapter.layer_calls else -1
        assert max_layer_called == 2, (
            f"Expected to stop at layer 2, but layers up to {max_layer_called} were called. "
            f"Layer calls: {adapter.layer_calls}"
        )


class TestGenerationTermination:
    """Test that generation terminates correctly."""

    def test_stops_on_eos_token_from_draft(self, mock_decoder):
        """Generation should stop when EOS token is produced during drafting."""
        # Return EOS token on first draft
        def mock_draft(input_ids, thresholds):
            return (mock_decoder.tokenizer.eos_token_id, 0, 2, 0.1)
        
        with patch.object(mock_decoder, '_draft_single_token', side_effect=mock_draft):
            input_ids = torch.tensor([[1, 2, 3]])
            thresholds = {0: 10.0, 1: 10.0}
            
            tokens = mock_decoder._generate_with_early_exit(
                input_ids, max_tokens=100, thresholds=thresholds
            )
        
        # Should stop immediately (0 tokens since EOS is not appended)
        assert len(tokens) == 0, f"Should stop on EOS, got {len(tokens)} tokens"

    def test_stops_at_max_tokens(self, mock_decoder):
        """Generation should stop at max_tokens limit."""
        num_layers = mock_decoder.adapter.get_num_layers()
        
        # Make draft return alternating early exit / lm_head tokens
        draft_count = [0]
        
        def mock_draft(input_ids, thresholds):
            draft_count[0] += 1
            # Alternate between early exit and lm_head to trigger verification
            if draft_count[0] % 2 == 1:
                return (10 + draft_count[0], 0, 2, 0.1)  # early exit
            else:
                return (20 + draft_count[0], None, num_layers, 0.0)  # lm_head
        
        def mock_model_forward(input_ids, **kwargs):
            seq_len = input_ids.shape[1]
            # Return logits that match the drafted tokens
            logits = torch.zeros(1, seq_len, 100)
            # Match all positions to their drafted values
            for pos in range(seq_len):
                expected_token = 10 + (pos + 1) if (pos + 1) % 2 == 1 else 20 + (pos + 1)
                logits[0, pos, expected_token % 100] = 100.0
            return MockOutput(logits)
        
        mock_decoder.model.set_forward(mock_model_forward)
        
        with patch.object(mock_decoder, '_draft_single_token', side_effect=mock_draft):
            input_ids = torch.tensor([[1, 2, 3]])
            thresholds = {0: 10.0, 1: 10.0}
            
            tokens = mock_decoder._generate_with_early_exit(
                input_ids, max_tokens=5, thresholds=thresholds
            )
        
        assert len(tokens) <= 5, f"Should stop at max_tokens=5, got {len(tokens)} tokens"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
