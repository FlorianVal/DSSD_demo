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
class StreamEvent:
    """Event for streaming generation updates."""

    event_type: str  # "draft", "verify_start", "accept", "reject", "full_model"
    tokens: List[TokenInfo]  # All tokens so far (validated)
    drafted_tokens: List[TokenInfo]  # Currently drafted (pending verification)
    message: str  # Human-readable status


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
        # Format prompt - check if tokenizer has a chat template set
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
                input_ids = self.tokenizer.encode(formatted, return_tensors="pt").to(
                    self.device
                )
            except Exception:
                # Fallback to raw prompt if chat template fails
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
                    self.device
                )
        else:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
                self.device
            )

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
        """
        # Format prompt
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
                input_ids = self.tokenizer.encode(formatted, return_tensors="pt").to(
                    self.device
                )
            except Exception:
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
                    self.device
                )
        else:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
                self.device
            )

        # Get thresholds
        thresholds = {}
        if self.calibration:
            thresholds = self.calibration.get_thresholds_for_level(accuracy_level)

        validated_tokens = []
        current_ids = input_ids.clone()
        num_layers = self.adapter.get_num_layers()
        head_layers = self.model_config.head_layer_indices

        while len(validated_tokens) < max_tokens:
            # ============================================================
            # DRAFT PHASE: Generate tokens using early exit heads
            # ============================================================
            drafted_tokens = []
            draft_ids = current_ids.clone()

            for _ in range(max_draft_length):
                if len(validated_tokens) + len(drafted_tokens) >= max_tokens:
                    break

                draft_result = self._draft_single_token(draft_ids, thresholds)

                if draft_result is None:
                    break

                token_id, exit_head, exit_layer, uncertainty = draft_result

                if token_id == self.tokenizer.eos_token_id:
                    break

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

                # Yield draft event
                yield StreamEvent(
                    event_type="draft",
                    tokens=list(validated_tokens),
                    drafted_tokens=list(drafted_tokens),
                    message=f"Drafting token {len(drafted_tokens)} using Head {exit_head}",
                )

            # ============================================================
            # VERIFY PHASE
            # ============================================================
            if drafted_tokens:
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
            else:
                # No drafts - generate with full model
                with torch.no_grad():
                    outputs = self.model(current_ids, use_cache=False)
                    logits = outputs.logits

                token_id = torch.argmax(logits[0, -1, :]).item()

                if token_id == self.tokenizer.eos_token_id:
                    break

                token_text = self.tokenizer.decode([token_id])
                full_token = TokenInfo(
                    token_id=token_id,
                    token_text=token_text,
                    exit_head=None,
                    exit_layer=num_layers,
                    uncertainty=0.0,
                )
                validated_tokens.append(full_token)
                current_ids = torch.cat(
                    [current_ids, torch.tensor([[token_id]], device=self.device)], dim=1
                )
                yield StreamEvent(
                    event_type="full_model",
                    tokens=list(validated_tokens),
                    drafted_tokens=[],
                    message=f"Full model: '{token_text}'",
                )

            if (
                validated_tokens
                and validated_tokens[-1].token_id == self.tokenizer.eos_token_id
            ):
                break

    def _generate_with_early_exit(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        thresholds: Dict[int, float],
        max_draft_length: int = 5,
    ) -> List[TokenInfo]:
        """
        Speculative decoding with early exit heads.

        GUARANTEES same output as full model by:
        1. DRAFT: Generate tokens using early exit heads (fast, partial compute)
        2. VERIFY: When full model needed, verify ALL drafted tokens
        3. ACCEPT: Keep matching tokens, take model's token at first mismatch
        """
        tokens = []
        current_ids = input_ids.clone()
        num_layers = self.adapter.get_num_layers()
        head_layers = self.model_config.head_layer_indices

        while len(tokens) < max_tokens:
            # ============================================================
            # DRAFT PHASE: Generate tokens using early exit heads
            # ============================================================
            drafted_tokens = []  # List of (token_id, exit_head, exit_layer, uncertainty)
            draft_ids = current_ids.clone()

            for _ in range(max_draft_length):
                if len(tokens) + len(drafted_tokens) >= max_tokens:
                    break

                # Try to draft a token using early exit
                draft_result = self._draft_single_token(draft_ids, thresholds)

                if draft_result is None:
                    # No head was confident enough - need to verify
                    break

                token_id, exit_head, exit_layer, uncertainty = draft_result

                if token_id == self.tokenizer.eos_token_id:
                    break

                drafted_tokens.append((token_id, exit_head, exit_layer, uncertainty))
                draft_ids = torch.cat(
                    [draft_ids, torch.tensor([[token_id]], device=self.device)], dim=1
                )

            # ============================================================
            # VERIFY PHASE: Run full model to verify drafted tokens
            # ============================================================
            if drafted_tokens:
                # Run full model on current_ids + all drafted tokens
                with torch.no_grad():
                    outputs = self.model(draft_ids, use_cache=False)
                    verify_logits = outputs.logits

                # Verify each drafted token
                start_pos = current_ids.shape[1] - 1  # Position before drafting

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
            else:
                # No tokens drafted - generate one with full model
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

            # Check for EOS in accepted tokens
            if tokens and tokens[-1].token_id == self.tokenizer.eos_token_id:
                break

        return tokens

    def _draft_single_token(
        self,
        input_ids: torch.Tensor,
        thresholds: Dict[int, float],
    ) -> Optional[Tuple[int, int, int, float]]:
        """
        Try to draft a single token using early exit heads.
        Returns (token_id, exit_head, exit_layer, uncertainty) if confident enough.
        Returns None if no head is confident enough (need full model verification).
        """
        device = input_ids.device
        seq_len = input_ids.shape[1]
        head_layers = self.model_config.head_layer_indices

        # Position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(
            0
        )

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

        # No head was confident enough - need full model verification
        return None

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
        """
        # Format prompt
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
                input_ids = self.tokenizer.encode(formatted, return_tensors="pt").to(
                    self.device
                )
            except Exception:
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
                    self.device
                )
        else:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
                self.device
            )

        tokens = []
        current_ids = input_ids.clone()
        num_layers = self.adapter.get_num_layers()

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
