# True Early Exit Inference with Dynamic Self-Speculative Decoding
# Provides actual speedup by stopping layer computation early

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
)

from .model_adapters import get_adapter, ModelAdapter
from .model_config import ModelConfig, CalibrationResult


def compute_entropy(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute entropy - lower = more confident."""
    probs = F.softmax(logits, dim=dim)
    log_probs = F.log_softmax(logits, dim=dim)
    return -torch.sum(probs * log_probs, dim=dim)


class AuxiliaryHead(nn.Module):
    """Auxiliary head for early exit prediction."""

    def __init__(
        self, hidden_size: int, vocab_size: int, norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        self.norm = norm_layer if norm_layer is not None else nn.Identity()
        self.linear = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(hidden_states))


@dataclass
class TokenInfo:
    """Information about a generated token for visualization."""

    token_id: int
    token_text: str
    exit_head: Optional[int]  # None = full model
    exit_layer: int
    uncertainty: float


@dataclass
class StreamingResult:
    """Result from streaming generation with accumulated metrics."""

    tokens: List[TokenInfo]
    total_time: float
    tokens_per_second: float
    avg_exit_layer: float
    exit_distribution: Dict[str, int]

    @classmethod
    def from_tokens(cls, tokens: List[TokenInfo], total_time: float, num_layers: int) -> "StreamingResult":
        """Build a StreamingResult from a list of tokens and timing info."""
        exit_dist: Dict[str, int] = {}
        layer_sum = 0

        for t in tokens:
            key = str(t.exit_head) if t.exit_head is not None else "full"
            exit_dist[key] = exit_dist.get(key, 0) + 1
            layer_sum += t.exit_layer

        avg_layer = layer_sum / len(tokens) if tokens else num_layers

        return cls(
            tokens=tokens,
            total_time=total_time,
            tokens_per_second=len(tokens) / total_time if total_time > 0 else 0,
            avg_exit_layer=avg_layer,
            exit_distribution=exit_dist,
        )


@dataclass
class StreamEvent:
    """Event for streaming generation updates."""

    event_type: str  # "draft", "verify_start", "accept", "reject", "full_model", "complete"
    tokens: List[TokenInfo]  # All tokens so far (validated)
    drafted_tokens: List[TokenInfo]  # Currently drafted (pending verification)
    message: str  # Human-readable status
    result: Optional[StreamingResult] = None  # Set on final "complete" event


@dataclass
class GenerationResult:
    """Complete generation result with token-level information."""

    text: str
    tokens: List[TokenInfo]
    total_time: float
    tokens_per_second: float
    avg_exit_layer: float
    exit_distribution: Dict[str, int]


class DSSDecoder:
    """
    Dynamic Self-Speculative Decoder with TRUE early exit.
    Actually stops computation at intermediate layers for speedup.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        adapter: ModelAdapter,
        aux_heads: nn.ModuleList,
        tokenizer: AutoTokenizer,
        model_config: ModelConfig,
        calibration: Optional[CalibrationResult] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.adapter = adapter
        self.aux_heads = aux_heads
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.calibration = calibration
        self.device = device
        self.uncertainty_fn = compute_entropy

    def _format_and_encode_prompt(self, prompt: str, use_chat_template: bool) -> torch.Tensor:
        """Format prompt with optional chat template and return input_ids tensor."""
        if (
            use_chat_template
            and hasattr(self.tokenizer, "chat_template")
            and self.tokenizer.chat_template is not None
        ):
            try:
                messages = [{"role": "user", "content": prompt}]
                formatted = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                return self.tokenizer.encode(formatted, return_tensors="pt").to(
                    self.device
                )
            except Exception:
                pass  # Fall through to raw prompt encoding
        return self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        use_early_exit: bool = True,
        accuracy_level: float = 0.75,
        use_chat_template: bool = True,
    ) -> GenerationResult:
        """
        Generate text with optional early exit.
        Returns detailed token-level information for visualization.
        """
        input_ids = self._format_and_encode_prompt(prompt, use_chat_template)

        # Get thresholds
        thresholds = {}
        if use_early_exit and self.calibration:
            thresholds = self.calibration.get_thresholds_for_level(accuracy_level)

        # Generate
        start_time = time.time()

        if use_early_exit:
            tokens = self._generate_with_early_exit(input_ids, max_tokens, thresholds)
        else:
            tokens = self._generate_full_model(input_ids, max_tokens)

        end_time = time.time()
        total_time = end_time - start_time

        # Build result
        text = "".join(t.token_text for t in tokens)
        exit_dist = defaultdict(int)
        layer_sum = 0

        for t in tokens:
            key = str(t.exit_head) if t.exit_head is not None else "full"
            exit_dist[key] += 1
            layer_sum += t.exit_layer

        avg_layer = (
            layer_sum / len(tokens) if tokens else self.model_config.num_hidden_layers
        )

        return GenerationResult(
            text=text,
            tokens=tokens,
            total_time=total_time,
            tokens_per_second=len(tokens) / total_time if total_time > 0 else 0,
            avg_exit_layer=avg_layer,
            exit_distribution=dict(exit_dist),
        )

    def generate_streaming(
        self,
        prompt: str,
        max_tokens: int = 100,
        accuracy_level: float = 0.75,
        use_chat_template: bool = True,
        max_draft_length: int = 5,
    ):
        """
        Generate with streaming - yields events showing draft/verify process.
        Each event shows current validated tokens and pending drafted tokens.
        Yields a final "complete" event with StreamingResult containing metrics.
        """
        input_ids = self._format_and_encode_prompt(prompt, use_chat_template)

        # Get thresholds
        thresholds = {}
        if self.calibration:
            thresholds = self.calibration.get_thresholds_for_level(accuracy_level)

        validated_tokens = []
        current_ids = input_ids.clone()
        num_layers = self.adapter.get_num_layers()
        start_time = time.time()

        while len(validated_tokens) < max_tokens:
            # ============================================================
            # DRAFT PHASE: Generate tokens using early exit or lm_head
            # ============================================================
            drafted_tokens = []
            draft_ids = current_ids.clone()
            got_lm_head_token = False
            should_stop = False

            for _ in range(max_draft_length):
                if len(validated_tokens) + len(drafted_tokens) >= max_tokens:
                    break

                # Generate a token (always returns a result)
                token_id, exit_head, exit_layer, uncertainty = self._draft_single_token(
                    draft_ids, thresholds
                )

                if token_id == self.tokenizer.eos_token_id:
                    # EOS handling
                    if exit_head is not None and drafted_tokens:
                        break  # Verify pending drafts first
                    should_stop = True
                    break  # Stop generation

                token_text = self.tokenizer.decode([token_id])
                drafted_token = TokenInfo(
                    token_id=token_id,
                    token_text=token_text,
                    exit_head=exit_head,
                    exit_layer=exit_layer,
                    uncertainty=uncertainty,
                )
                drafted_tokens.append(drafted_token)
                draft_ids = torch.cat(
                    [draft_ids, torch.tensor([[token_id]], device=self.device)], dim=1
                )

                if exit_head is None:
                    # Token from lm_head - triggers verification
                    got_lm_head_token = True
                    yield StreamEvent(
                        event_type="draft",
                        tokens=list(validated_tokens),
                        drafted_tokens=list(drafted_tokens),
                        message=f"Drafting token {len(drafted_tokens)} using Full Model",
                    )
                    break
                else:
                    # Token from early exit head
                    yield StreamEvent(
                        event_type="draft",
                        tokens=list(validated_tokens),
                        drafted_tokens=list(drafted_tokens),
                        message=f"Drafting token {len(drafted_tokens)} using Head {exit_head}",
                    )

            # Check if we should stop (EOS encountered with no pending drafts)
            if should_stop:
                break

            # ============================================================
            # VERIFY PHASE
            # ============================================================
            if not drafted_tokens:
                break

            yield StreamEvent(
                event_type="verify_start",
                tokens=list(validated_tokens),
                drafted_tokens=list(drafted_tokens),
                message=f"Verifying {len(drafted_tokens)} drafted tokens...",
            )

            with torch.no_grad():
                outputs = self.model(draft_ids, use_cache=False)
                verify_logits = outputs.logits

            start_pos = current_ids.shape[1] - 1
            all_accepted = True

            for i, drafted_token in enumerate(drafted_tokens):
                verify_pos = start_pos + i
                verified_token_id = torch.argmax(
                    verify_logits[0, verify_pos, :]
                ).item()

                if drafted_token.token_id == verified_token_id:
                    # Accept
                    validated_tokens.append(drafted_token)
                    current_ids = torch.cat(
                        [
                            current_ids,
                            torch.tensor(
                                [[drafted_token.token_id]], device=self.device
                            ),
                        ],
                        dim=1,
                    )
                    yield StreamEvent(
                        event_type="accept",
                        tokens=list(validated_tokens),
                        drafted_tokens=[],
                        message=f"✓ Accepted '{drafted_token.token_text}'",
                    )
                else:
                    # Reject - use full model's token
                    all_accepted = False
                    token_text = self.tokenizer.decode([verified_token_id])
                    corrected_token = TokenInfo(
                        token_id=verified_token_id,
                        token_text=token_text,
                        exit_head=None,
                        exit_layer=num_layers,
                        uncertainty=0.0,
                    )
                    validated_tokens.append(corrected_token)
                    current_ids = torch.cat(
                        [
                            current_ids,
                            torch.tensor([[verified_token_id]], device=self.device),
                        ],
                        dim=1,
                    )
                    yield StreamEvent(
                        event_type="reject",
                        tokens=list(validated_tokens),
                        drafted_tokens=[],
                        message=f"✗ Rejected '{drafted_token.token_text}' → '{token_text}'",
                    )
                    break

            # BONUS TOKEN: If all tokens were accepted, get bonus from last position
            if all_accepted and len(validated_tokens) < max_tokens:
                bonus_pos = start_pos + len(drafted_tokens)
                if bonus_pos < verify_logits.shape[1]:
                    bonus_token_id = torch.argmax(
                        verify_logits[0, bonus_pos, :]
                    ).item()
                    if bonus_token_id != self.tokenizer.eos_token_id:
                        bonus_text = self.tokenizer.decode([bonus_token_id])
                        bonus_token = TokenInfo(
                            token_id=bonus_token_id,
                            token_text=bonus_text,
                            exit_head=None,
                            exit_layer=num_layers,
                            uncertainty=0.0,
                        )
                        validated_tokens.append(bonus_token)
                        current_ids = torch.cat(
                            [
                                current_ids,
                                torch.tensor([[bonus_token_id]], device=self.device),
                            ],
                            dim=1,
                        )
                        yield StreamEvent(
                            event_type="accept",
                            tokens=list(validated_tokens),
                            drafted_tokens=[],
                            message=f"✓ Bonus token '{bonus_text}'",
                        )

            if (
                validated_tokens
                and validated_tokens[-1].token_id == self.tokenizer.eos_token_id
            ):
                break

        # Yield final "complete" event with metrics
        total_time = time.time() - start_time
        result = StreamingResult.from_tokens(validated_tokens, total_time, num_layers)
        yield StreamEvent(
            event_type="complete",
            tokens=list(validated_tokens),
            drafted_tokens=[],
            message="Generation complete",
            result=result,
        )

    def _generate_with_early_exit(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        thresholds: Dict[int, float],
        max_draft_length: int = 5,
    ) -> List[TokenInfo]:
        """
        Speculative decoding with early exit heads.

        The flow:
        1. Generate tokens using _draft_single_token (which may early exit or use lm_head)
        2. Tokens from early exit heads are "drafts" that need verification
        3. When we get a token from lm_head (exit_head=None), it triggers verification
           of all pending drafts, and the lm_head token is accepted as verified
        4. All accepted tokens are guaranteed to match full model output
        """
        tokens = []
        current_ids = input_ids.clone()
        num_layers = self.adapter.get_num_layers()

        while len(tokens) < max_tokens:
            # ============================================================
            # DRAFT PHASE: Generate tokens, collecting early exit drafts
            # ============================================================
            drafted_tokens = []  # List of (token_id, exit_head, exit_layer, uncertainty)
            draft_ids = current_ids.clone()
            got_lm_head_token = False

            for _ in range(max_draft_length):
                if len(tokens) + len(drafted_tokens) >= max_tokens:
                    break

                # Generate a token (always returns a result, never None)
                token_id, exit_head, exit_layer, uncertainty = self._draft_single_token(
                    draft_ids, thresholds
                )

                if token_id == self.tokenizer.eos_token_id:
                    # If EOS from early exit, we still need to verify pending drafts
                    if exit_head is not None and drafted_tokens:
                        # Don't add EOS to drafts, just break to verify
                        break
                    # If EOS from lm_head or no pending drafts, we're done
                    return tokens

                if exit_head is None:
                    # Token from lm_head - this is verified, triggers verification of drafts
                    got_lm_head_token = True
                    # Add to drafts for unified handling, but mark as already verified
                    drafted_tokens.append((token_id, exit_head, exit_layer, uncertainty))
                    draft_ids = torch.cat(
                        [draft_ids, torch.tensor([[token_id]], device=self.device)], dim=1
                    )
                    break  # Stop drafting, go to verification
                else:
                    # Token from early exit head - add to drafts for later verification
                    drafted_tokens.append((token_id, exit_head, exit_layer, uncertainty))
                    draft_ids = torch.cat(
                        [draft_ids, torch.tensor([[token_id]], device=self.device)], dim=1
                    )

            # ============================================================
            # VERIFY PHASE: Verify drafted tokens with full model
            # ============================================================
            if not drafted_tokens:
                # No tokens generated (shouldn't happen with the new logic)
                break

            # If the last token is from lm_head, we already have full model output
            # for all positions. Use it for verification.
            last_token = drafted_tokens[-1]
            _, last_exit_head, _, _ = last_token

            if last_exit_head is None:
                # Last token is from lm_head - all earlier tokens need verification
                # The lm_head pass already computed logits for all positions
                # We can use the model output to verify
                
                # Need to run full model to get logits for verification
                with torch.no_grad():
                    outputs = self.model(draft_ids, use_cache=False)
                    verify_logits = outputs.logits

                start_pos = current_ids.shape[1] - 1

                for i, (drafted_token, exit_head, exit_layer, uncertainty) in enumerate(
                    drafted_tokens
                ):
                    verify_pos = start_pos + i
                    verified_token = torch.argmax(
                        verify_logits[0, verify_pos, :]
                    ).item()

                    if drafted_token == verified_token:
                        # Token matches - accept it
                        token_text = self.tokenizer.decode([drafted_token])
                        tokens.append(
                            TokenInfo(
                                token_id=drafted_token,
                                token_text=token_text,
                                exit_head=exit_head,
                                exit_layer=exit_layer,
                                uncertainty=uncertainty,
                            )
                        )
                        current_ids = torch.cat(
                            [
                                current_ids,
                                torch.tensor([[drafted_token]], device=self.device),
                            ],
                            dim=1,
                        )
                    else:
                        # Mismatch - use full model's token
                        token_text = self.tokenizer.decode([verified_token])
                        tokens.append(
                            TokenInfo(
                                token_id=verified_token,
                                token_text=token_text,
                                exit_head=None,  # Full model
                                exit_layer=num_layers,
                                uncertainty=0.0,
                            )
                        )
                        current_ids = torch.cat(
                            [
                                current_ids,
                                torch.tensor([[verified_token]], device=self.device),
                            ],
                            dim=1,
                        )
                        # Stop - discard remaining drafted tokens
                        break

                # BONUS TOKEN: If all drafted tokens were accepted, use the last position
                # to get an additional token (this is the "free" token from lm_head)
                if len(tokens) >= len(drafted_tokens):
                    # All drafts were accepted, check for bonus token
                    bonus_pos = start_pos + len(drafted_tokens)
                    if bonus_pos < verify_logits.shape[1]:
                        bonus_token_id = torch.argmax(
                            verify_logits[0, bonus_pos, :]
                        ).item()
                        if (
                            bonus_token_id != self.tokenizer.eos_token_id
                            and len(tokens) < max_tokens
                        ):
                            bonus_text = self.tokenizer.decode([bonus_token_id])
                            tokens.append(
                                TokenInfo(
                                    token_id=bonus_token_id,
                                    token_text=bonus_text,
                                    exit_head=None,  # Full model
                                    exit_layer=num_layers,
                                    uncertainty=0.0,
                                )
                            )
                            current_ids = torch.cat(
                                [
                                    current_ids,
                                    torch.tensor(
                                        [[bonus_token_id]], device=self.device
                                    ),
                                ],
                                dim=1,
                            )
            else:
                # All tokens are from early exit heads - need to run full model for verification
                with torch.no_grad():
                    outputs = self.model(draft_ids, use_cache=False)
                    verify_logits = outputs.logits

                start_pos = current_ids.shape[1] - 1

                for i, (drafted_token, exit_head, exit_layer, uncertainty) in enumerate(
                    drafted_tokens
                ):
                    verify_pos = start_pos + i
                    verified_token = torch.argmax(
                        verify_logits[0, verify_pos, :]
                    ).item()

                    if drafted_token == verified_token:
                        # Token matches - accept it with early exit info
                        token_text = self.tokenizer.decode([drafted_token])
                        tokens.append(
                            TokenInfo(
                                token_id=drafted_token,
                                token_text=token_text,
                                exit_head=exit_head,
                                exit_layer=exit_layer,
                                uncertainty=uncertainty,
                            )
                        )
                        current_ids = torch.cat(
                            [
                                current_ids,
                                torch.tensor([[drafted_token]], device=self.device),
                            ],
                            dim=1,
                        )
                    else:
                        # Mismatch - use full model's token
                        token_text = self.tokenizer.decode([verified_token])
                        tokens.append(
                            TokenInfo(
                                token_id=verified_token,
                                token_text=token_text,
                                exit_head=None,  # Full model
                                exit_layer=num_layers,
                                uncertainty=0.0,
                            )
                        )
                        current_ids = torch.cat(
                            [
                                current_ids,
                                torch.tensor([[verified_token]], device=self.device),
                            ],
                            dim=1,
                        )
                        # Stop - discard remaining drafted tokens
                        break

                # BONUS TOKEN from verification pass
                if len(tokens) >= len(drafted_tokens):
                    bonus_pos = start_pos + len(drafted_tokens)
                    if bonus_pos < verify_logits.shape[1]:
                        bonus_token_id = torch.argmax(
                            verify_logits[0, bonus_pos, :]
                        ).item()
                        if (
                            bonus_token_id != self.tokenizer.eos_token_id
                            and len(tokens) < max_tokens
                        ):
                            bonus_text = self.tokenizer.decode([bonus_token_id])
                            tokens.append(
                                TokenInfo(
                                    token_id=bonus_token_id,
                                    token_text=bonus_text,
                                    exit_head=None,  # Full model
                                    exit_layer=num_layers,
                                    uncertainty=0.0,
                                )
                            )
                            current_ids = torch.cat(
                                [
                                    current_ids,
                                    torch.tensor(
                                        [[bonus_token_id]], device=self.device
                                    ),
                                ],
                                dim=1,
                            )

            # Check for EOS in accepted tokens
            if tokens and tokens[-1].token_id == self.tokenizer.eos_token_id:
                break

        return tokens

    def _draft_single_token(
        self,
        input_ids: torch.Tensor,
        thresholds: Dict[int, float],
    ) -> Tuple[int, Optional[int], int, float]:
        """
        Generate a single token using early exit or full model.
        
        Returns (token_id, exit_head, exit_layer, uncertainty):
        - If an early exit head is confident: returns token with that head's info
        - If no head is confident: continues to lm_head and returns token from there
        
        This function ALWAYS returns a token (never returns None).
        """
        device = input_ids.device
        seq_len = input_ids.shape[1]
        head_layers = self.model_config.head_layer_indices
        num_layers = self.adapter.get_num_layers()

        # Position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(
            0
        )

        # Cache position (required by newer transformers for Qwen3)
        cache_position = torch.arange(seq_len, dtype=torch.long, device=device)

        # Get embeddings
        hidden_states = self.adapter.get_embed_tokens(input_ids)

        # Get rotary embeddings
        position_embeddings = self.adapter.get_position_embeddings(
            hidden_states, position_ids
        )

        # Sort heads by layer
        sorted_heads = sorted(enumerate(head_layers), key=lambda x: x[1])

        # Iterate through layers
        with torch.no_grad():
            for layer_idx, layer in enumerate(self.adapter.get_layers()):
                hidden_states, _ = self.adapter.forward_layer(
                    layer=layer,
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    attention_mask=None,
                    past_key_value=None,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                    cache_position=cache_position,
                )

                # Check if this is a head checkpoint
                for head_idx, head_layer in sorted_heads:
                    if layer_idx == head_layer:
                        # Run aux head on last position
                        aux_head = self.aux_heads[head_idx]
                        head_device = next(aux_head.parameters()).device
                        head_input = hidden_states[:, -1:, :].to(head_device)
                        head_logits = aux_head(head_input)
                        uncertainty = self.uncertainty_fn(
                            head_logits[:, -1, :], dim=-1
                        ).item()

                        # Check threshold - if confident, return drafted token
                        if (
                            head_idx in thresholds
                            and uncertainty < thresholds[head_idx]
                        ):
                            token_id = torch.argmax(head_logits[0, -1, :]).item()
                            return (token_id, head_idx, layer_idx, uncertainty)

            # No head was confident - use lm_head to get the token
            # Apply final norm and lm_head
            final_hidden = self.adapter.apply_final_norm(hidden_states)
            logits = self.adapter.get_lm_head_output(final_hidden)
            
            # Get token from last position
            token_id = torch.argmax(logits[0, -1, :]).item()
            
            # Compute uncertainty for the lm_head output
            uncertainty = self.uncertainty_fn(logits[0, -1, :].unsqueeze(0), dim=-1).item()
            
            return (token_id, None, num_layers, uncertainty)

    def _generate_full_model(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
    ) -> List[TokenInfo]:
        """Generate using full model (no early exit)."""
        tokens = []
        current_ids = input_ids.clone()
        num_layers = self.adapter.get_num_layers()

        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(current_ids, use_cache=False)
                logits = outputs.logits

            token_id = torch.argmax(logits[0, -1, :]).item()

            if token_id == self.tokenizer.eos_token_id:
                break

            token_text = self.tokenizer.decode([token_id])
            tokens.append(
                TokenInfo(
                    token_id=token_id,
                    token_text=token_text,
                    exit_head=None,
                    exit_layer=num_layers,
                    uncertainty=0.0,
                )
            )

            current_ids = torch.cat(
                [current_ids, torch.tensor([[token_id]], device=self.device)], dim=1
            )

        return tokens

    def generate_full_model_streaming(
        self,
        prompt: str,
        max_tokens: int = 100,
        use_chat_template: bool = True,
    ):
        """
        Generate with full model in streaming mode - yields each token as generated.
        Yields a final "complete" event with StreamingResult containing metrics.
        """
        input_ids = self._format_and_encode_prompt(prompt, use_chat_template)

        tokens = []
        current_ids = input_ids.clone()
        num_layers = self.adapter.get_num_layers()
        start_time = time.time()

        for i in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(current_ids, use_cache=False)
                logits = outputs.logits

            token_id = torch.argmax(logits[0, -1, :]).item()

            if token_id == self.tokenizer.eos_token_id:
                break

            token_text = self.tokenizer.decode([token_id])
            token_info = TokenInfo(
                token_id=token_id,
                token_text=token_text,
                exit_head=None,
                exit_layer=num_layers,
                uncertainty=0.0,
            )
            tokens.append(token_info)

            current_ids = torch.cat(
                [current_ids, torch.tensor([[token_id]], device=self.device)], dim=1
            )

            yield StreamEvent(
                event_type="full_model",
                tokens=list(tokens),
                drafted_tokens=[],
                message=f"Token {i + 1}: '{token_text}'",
            )

        # Yield final "complete" event with metrics
        total_time = time.time() - start_time
        result = StreamingResult.from_tokens(tokens, total_time, num_layers)
        yield StreamEvent(
            event_type="complete",
            tokens=list(tokens),
            drafted_tokens=[],
            message="Generation complete",
            result=result,
        )


def load_dssd_model(
    model_name: str,
    heads_path: str,
    config_path: str,
    calibration_path: Optional[str] = None,
    device: str = "auto",
) -> Tuple[DSSDecoder, AutoTokenizer]:
    """
    Load a DSSD model from HuggingFace Hub or local paths.

    Args:
        model_name: HuggingFace model name (e.g., "meta-llama/Meta-Llama-3-8B")
        heads_path: Path to aux_heads.pt
        config_path: Path to config.json
        calibration_path: Optional path to calibration.json
        device: Device to load on

    Returns:
        decoder: DSSDecoder ready for generation
        tokenizer: Tokenizer for the model
    """
    # Load config
    model_config = ModelConfig.from_json(config_path)

    # Load calibration if provided
    calibration = None
    if calibration_path:
        calibration = CalibrationResult.from_json(calibration_path)

    # Quantization config
    quant_config = None
    if model_config.quantization == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float32,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif model_config.quantization == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map=device,
    )
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get adapter
    adapter = get_adapter(model)

    # Determine the norm type and create aux heads WITHOUT deepcopy (to avoid accelerate hooks)
    aux_heads = nn.ModuleList()

    # Get norm config from model
    norm_eps = 1e-6
    if hasattr(model.config, "rms_norm_eps"):
        norm_eps = model.config.rms_norm_eps
    elif hasattr(model.config, "layer_norm_eps"):
        norm_eps = model.config.layer_norm_eps

    for _ in range(model_config.num_heads):
        # Create fresh RMSNorm (or LayerNorm) without accelerate hooks
        norm_layer = nn.RMSNorm(model_config.hidden_size, eps=norm_eps)

        head = AuxiliaryHead(
            model_config.hidden_size,
            model_config.vocab_size,
            norm_layer,
        )
        aux_heads.append(head)

    # Load trained weights (this will properly set the norm weights)
    state_dict = torch.load(heads_path, map_location="cpu")
    aux_heads.load_state_dict(state_dict)

    # Move to device - use cuda:0 to keep on single device
    model_device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    model_dtype = next(model.parameters()).dtype
    aux_heads = aux_heads.to(device=model_device, dtype=model_dtype)
    aux_heads.eval()

    # Create decoder
    decoder = DSSDecoder(
        model=model,
        adapter=adapter,
        aux_heads=aux_heads,
        tokenizer=tokenizer,
        model_config=model_config,
        calibration=calibration,
        device=str(model_device),
    )

    return decoder, tokenizer
