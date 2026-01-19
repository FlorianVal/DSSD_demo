"""
DSSD Demo - Dynamic Self-Speculative Decoding Visualization
Showcases early exit inference with color-coded tokens showing which head generated each token.
"""

import gradio as gr
from dataclasses import dataclass
from pathlib import Path
import time
from huggingface_hub import hf_hub_download

from src.inference import load_dssd_model, DSSDecoder, TokenInfo, StreamEvent, StreamingResult

# Available models configuration
AVAILABLE_MODELS = {
    "DSSD-Llama3-8B": {
        "model_name": "meta-llama/Meta-Llama-3-8B",
        "repo_id": "valcore/DSSD-Llama3-8B",
        "local_path": "../checkpoints/llama3-8b-4bit",
    },
    "DSSD-Qwen3-0.6B": {
        "model_name": "Qwen/Qwen3-0.6B",
        "repo_id": "valcore/DSSD-Qwen3-0.6B",
        "local_path": "../checkpoints/qwen3-0.6b",
    },
}

# Color palette for exit heads (colorblind-friendly)
HEAD_COLORS = [
    "#E63946",  # Red - Head 0 (earliest)
    "#F4A261",  # Orange - Head 1
    "#2A9D8F",  # Teal - Head 2
    "#457B9D",  # Blue - Head 3
    "#8338EC",  # Purple - Head 4
]
FULL_MODEL_COLOR = "#95D5B2"  # Light green - Full model

PENDING_TOKEN_BORDER = "var(--border-color-primary)"
PENDING_TOKEN_TEXT = "var(--body-text-color)"
DRAFTED_FALLBACK_COLOR = "var(--neutral-200)"

# Global decoder cache
_decoder_cache = {}


def get_decoder(model_key: str) -> DSSDecoder:
    """Get or load a decoder for the specified model."""
    global _decoder_cache

    if model_key in _decoder_cache:
        return _decoder_cache[model_key]

    model_info = AVAILABLE_MODELS[model_key]

    # Try local path first (for development)
    local_dir = Path(__file__).parent / model_info["local_path"]
    heads_path = local_dir / "aux_heads.pt"
    config_path = local_dir / "config.json"
    calibration_path = local_dir / "calibration.json"

    if heads_path.exists() and config_path.exists():
        print(f"Loading model heads from local path: {local_dir}")
        # calibration_path is optional, so no need to check its existence here
    else:
        # Download from HF Hub
        repo_id = model_info["repo_id"]
        print(f"Downloading model heads from {repo_id}...")
        heads_path = hf_hub_download(repo_id=repo_id, filename="aux_heads.pt")
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        try:
            calibration_path = hf_hub_download(
                repo_id=repo_id, filename="calibration.json"
            )
        except Exception:
            calibration_path = None  # calibration.json is optional

    decoder, tokenizer = load_dssd_model(
        model_name=model_info["model_name"],
        heads_path=str(heads_path),
        config_path=str(config_path),
        calibration_path=str(calibration_path) if calibration_path else None,
        device="auto",
    )

    _decoder_cache[model_key] = decoder
    return decoder


def tokens_to_html(tokens: list[TokenInfo], head_layers: list[int]) -> str:
    """Convert token info list to color-coded HTML."""
    html_parts = []

    for token in tokens:
        if token.exit_head is not None:
            color = HEAD_COLORS[token.exit_head % len(HEAD_COLORS)]
            layer = head_layers[token.exit_head]
            title = f"Head {token.exit_head} (Layer {layer})"
        else:
            color = FULL_MODEL_COLOR
            title = f"Full Model (Layer {token.exit_layer})"

        # Escape HTML special chars
        text = (
            token.token_text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        text = text.replace("\n", "<br>").replace(" ", "&nbsp;")

        html_parts.append(
            f'<span style="background-color: {color}; padding: 2px 4px; '
            f'border-radius: 3px; margin: 1px; display: inline-block; color: #111827;" title="{title}">{text}</span>'
        )

    # Wrap in container with word-wrap to prevent overflow
    tokens_html = "".join(html_parts)
    return f"""<div style="word-wrap: break-word; overflow-wrap: break-word; max-width: 100%; line-height: 1.8;">{tokens_html}</div>"""


def drafted_tokens_to_html(tokens: list[TokenInfo], head_layers: list[int]) -> str:
    """Convert drafted (pending) tokens to HTML with dashed border style."""
    html_parts = []

    for token in tokens:
        if token.exit_head is not None:
            color = HEAD_COLORS[token.exit_head % len(HEAD_COLORS)]
            layer = head_layers[token.exit_head]
            title = f"PENDING - Head {token.exit_head} (Layer {layer})"
        else:
            color = DRAFTED_FALLBACK_COLOR
            title = "PENDING - Unassigned"

        text = (
            token.token_text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        text = text.replace("\n", "<br>").replace(" ", "&nbsp;")

        html_parts.append(
            f'<span style="background-color: {color}; padding: 2px 4px; '
            f"border-radius: 3px; margin: 1px; display: inline-block; "
            f"border: 2px dashed {PENDING_TOKEN_BORDER}; color: {PENDING_TOKEN_TEXT}; "
            f'opacity: 0.75;" title="{title}">{text}</span>'
        )

    return "".join(html_parts)


def create_legend(head_layers: list[int]) -> str:
    """Create HTML legend for the color scheme."""
    legend_items = []
    for i, layer in enumerate(head_layers):
        color = HEAD_COLORS[i % len(HEAD_COLORS)]
        legend_items.append(
            f'<span style="background-color: {color}; padding: 4px 8px; '
            f'border-radius: 4px; margin-right: 8px;">Head {i} (Layer {layer})</span>'
        )
    legend_items.append(
        f'<span style="background-color: {FULL_MODEL_COLOR}; padding: 4px 8px; '
        f'border-radius: 4px;">Full Model</span>'
    )
    return " ".join(legend_items)



@dataclass
class StatsPayload:
    generated_at: float
    speedup_text: str
    ee_time: str | None
    ee_tps: str | None
    ee_avg: str | None
    full_time: str | None
    full_tps: str | None
    full_avg: str | None
    show_ee: bool
    show_full: bool


def build_stats_outputs(
    result_ee,
    result_full,
    use_early_exit: bool,
    compare_mode: bool,
    generated_at: float | None = None,
):
    speedup_text = ""
    if result_ee and result_full and result_full.tokens_per_second > 0:
        speedup = result_ee.tokens_per_second / result_full.tokens_per_second
        speedup_text = f"**Speedup:** {speedup:.2f}x"
    elif result_ee:
        speedup_text = "**Speedup:** N/A (full model not run)"
    elif result_full:
        speedup_text = "**Speedup:** N/A (early exit disabled)"

    if not speedup_text:
        speedup_text = "**Speedup:** N/A"

    ee_time = f"{result_ee.total_time:.2f}" if result_ee else None
    ee_tps = f"{result_ee.tokens_per_second:.2f}" if result_ee else None
    ee_avg = f"{result_ee.avg_exit_layer:.1f}" if result_ee else None

    full_time = f"{result_full.total_time:.2f}" if result_full else None
    full_tps = f"{result_full.tokens_per_second:.2f}" if result_full else None
    full_avg = f"{result_full.avg_exit_layer:.1f}" if result_full else None

    show_ee = compare_mode or use_early_exit
    show_full = compare_mode or not use_early_exit

    return StatsPayload(
        generated_at=generated_at if generated_at is not None else time.time(),
        speedup_text=speedup_text,
        ee_time=ee_time,
        ee_tps=ee_tps,
        ee_avg=ee_avg,
        full_time=full_time,
        full_tps=full_tps,
        full_avg=full_avg,
        show_ee=show_ee,
        show_full=show_full,
    )


def stats_payload_to_outputs(payload: StatsPayload):
    return (
        payload.speedup_text,
        payload.ee_time,
        payload.ee_tps,
        payload.ee_avg,
        payload.full_time,
        payload.full_tps,
        payload.full_avg,
        gr.update(visible=payload.show_ee),
        gr.update(visible=payload.show_full),
    )



def generate(
    prompt: str,
    model_key: str,
    use_early_exit: bool,
    accuracy_level: float,
    max_tokens: int,
    compare_mode: bool,
):
    """Main generation function for Gradio interface with streaming."""
    initial_stats_timestamp = time.time()
    try:
        decoder = get_decoder(model_key)
    except Exception as e:
        error_msg = f"<p style='color: red;'>Error loading model: {e}</p>"
        status_msg = f"**Error loading model:** {e}"
        stats_payload = build_stats_outputs(
            None,
            None,
            use_early_exit,
            compare_mode,
            generated_at=initial_stats_timestamp,
        )
        yield (
            error_msg,
            "",
            status_msg,
            *stats_payload_to_outputs(stats_payload),
            "",
        )
        return


    head_layers = decoder.model_config.head_layer_indices
    legend = create_legend(head_layers)

    # Get calibration accuracy levels
    if decoder.calibration:
        available_levels = decoder.calibration.accuracy_levels
        closest_level = min(available_levels, key=lambda x: abs(x - accuracy_level))
    else:
        closest_level = accuracy_level

    if compare_mode:
        # Compare mode with streaming for early exit
        # First, stream the early exit generation
        final_ee_tokens = []
        ee_streaming_result = None

        for event in decoder.generate_streaming(
            prompt=prompt,
            max_tokens=int(max_tokens),
            accuracy_level=closest_level,
            use_chat_template=True,
        ):
            # Handle "complete" event - extract result and break
            if event.event_type == "complete":
                ee_streaming_result = event.result
                final_ee_tokens = event.tokens
                break

            final_ee_tokens = event.tokens
            validated_html = ""
            if event.tokens:
                validated_html = tokens_to_html(event.tokens, head_layers)
                validated_html = validated_html.replace(
                    '<div style="word-wrap: break-word; overflow-wrap: break-word; max-width: 100%; line-height: 1.8;">',
                    "",
                ).rstrip("</div>")

            drafted_html = ""
            if event.drafted_tokens:
                drafted_html = drafted_tokens_to_html(event.drafted_tokens, head_layers)

            combined_html = f"""<div style="word-wrap: break-word; overflow-wrap: break-word; max-width: 100%; line-height: 1.8;">{validated_html}{drafted_html}</div>"""

            status = (
                "**Early Exit:** {message}  \n"
                "**Full Model:** Waiting..."
            ).format(
                message=event.message,
            )

            stats_payload = build_stats_outputs(
                None,
                None,
                use_early_exit,
                compare_mode,
                generated_at=initial_stats_timestamp,
            )
            yield (
                combined_html,
                "<p style='color: var(--body-text-color-subdued);'>Waiting for early exit to complete...</p>",
                status,
                *stats_payload_to_outputs(stats_payload),
                legend,
            )

        # Now stream full model
        final_full_tokens = []
        full_streaming_result = None

        for event in decoder.generate_full_model_streaming(
            prompt=prompt,
            max_tokens=int(max_tokens),
            use_chat_template=True,
        ):
            # Handle "complete" event - extract result and break
            if event.event_type == "complete":
                full_streaming_result = event.result
                final_full_tokens = event.tokens
                break

            final_full_tokens = event.tokens
            html_full = tokens_to_html(event.tokens, head_layers)
            status = (
                "**Full Model:** {message}"
            ).format(
                message=event.message,
            )
            stats_payload = build_stats_outputs(
                None,
                None,
                use_early_exit,
                compare_mode,
                generated_at=initial_stats_timestamp,
            )
            yield (
                tokens_to_html(final_ee_tokens, head_layers),
                html_full,
                status,
                *stats_payload_to_outputs(stats_payload),
                legend,
            )

        # Final output with metrics from streaming results (no re-run needed)
        html_ee = tokens_to_html(final_ee_tokens, head_layers)
        html_full = tokens_to_html(final_full_tokens, head_layers)

        stats_payload = build_stats_outputs(ee_streaming_result, full_streaming_result, use_early_exit, compare_mode)
        yield (
            html_ee,
            html_full,
            "",
            *stats_payload_to_outputs(stats_payload),
            legend,
        )

    elif use_early_exit:
        # STREAMING mode for early exit - show draft/verify process
        streaming_result = None
        final_tokens = []

        for event in decoder.generate_streaming(
            prompt=prompt,
            max_tokens=int(max_tokens),
            accuracy_level=closest_level,
            use_chat_template=True,
        ):
            # Handle "complete" event - extract result and break
            if event.event_type == "complete":
                streaming_result = event.result
                final_tokens = event.tokens
                break

            final_tokens = event.tokens

            # Build HTML showing validated + drafted tokens
            validated_html = ""
            if event.tokens:
                validated_html = tokens_to_html(event.tokens, head_layers)
                # Remove the outer div to combine with drafted
                validated_html = validated_html.replace(
                    '<div style="word-wrap: break-word; overflow-wrap: break-word; max-width: 100%; line-height: 1.8;">',
                    "",
                ).rstrip("</div>")

            drafted_html = ""
            if event.drafted_tokens:
                drafted_html = drafted_tokens_to_html(event.drafted_tokens, head_layers)

            # Combine
            combined_html = f"""<div style="word-wrap: break-word; overflow-wrap: break-word; max-width: 100%; line-height: 1.8;">{validated_html}{drafted_html}</div>"""

            # Status message
            status = (
                "**Status:** {message}"
            ).format(
                message=event.message,
            )

            stats_payload = build_stats_outputs(
                None,
                None,
                use_early_exit,
                compare_mode,
                generated_at=initial_stats_timestamp,
            )
            yield (
                combined_html,
                "",
                status,
                *stats_payload_to_outputs(stats_payload),
                legend,
            )

        # Final output with metrics from streaming result (no re-run needed)
        html = tokens_to_html(final_tokens, head_layers)
        stats_payload = build_stats_outputs(streaming_result, None, use_early_exit, compare_mode)
        yield (
            html,
            "",
            "",
            *stats_payload_to_outputs(stats_payload),
            legend,
        )

    else:
        # Full model mode (streaming)
        streaming_result = None
        final_tokens = []

        for event in decoder.generate_full_model_streaming(
            prompt=prompt,
            max_tokens=int(max_tokens),
            use_chat_template=True,
        ):
            # Handle "complete" event - extract result and break
            if event.event_type == "complete":
                streaming_result = event.result
                final_tokens = event.tokens
                break

            final_tokens = event.tokens
            html = tokens_to_html(event.tokens, head_layers)
            status = (
                "**Full Model:** {message}"
            ).format(
                message=event.message,
            )
            stats_payload = build_stats_outputs(
                None,
                None,
                use_early_exit,
                compare_mode,
                generated_at=initial_stats_timestamp,
            )
            yield (
                html,
                "",
                status,
                *stats_payload_to_outputs(stats_payload),
                legend,
            )

        # Final output with metrics from streaming result (no re-run needed)
        html = tokens_to_html(final_tokens, head_layers)
        stats_payload = build_stats_outputs(None, streaming_result, use_early_exit, compare_mode)
        yield (
            html,
            "",
            "",
            *stats_payload_to_outputs(stats_payload),
            legend,
        )


def build_demo():
    """Build the Gradio demo interface."""
    with gr.Blocks(title="DSSD Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🚀 Dynamic Self-Speculative Decoding (DSSD) Demo
        
        This demo showcases **early exit inference** where tokens can be generated from intermediate 
        layers when the model is confident, resulting in faster generation.
        
        **Colors indicate which layer generated each token** - earlier layers = faster!
        """)

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=3,
                    value="What is machine learning in simple terms?",
                )

                model_selector = gr.Dropdown(
                    label="Model",
                    choices=list(AVAILABLE_MODELS.keys()),
                    value=list(AVAILABLE_MODELS.keys())[0],
                )

                with gr.Row():
                    use_early_exit = gr.Checkbox(label="Enable Early Exit", value=True)
                    compare_mode = gr.Checkbox(label="Compare Mode", value=False)

                accuracy_level = gr.Slider(
                    label="Accuracy Level",
                    minimum=0.6,
                    maximum=0.99,
                    step=0.05,
                    value=0.75,
                    info="Higher = more accurate but slower",
                )

                max_tokens = gr.Slider(
                    label="Max Tokens",
                    minimum=10,
                    maximum=200,
                    step=10,
                    value=50,
                )

                generate_btn = gr.Button("Generate", variant="primary")

        # Legend (full width, above outputs)
        legend_html = gr.HTML()

        # Outputs section - dynamic based on compare mode
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Generated Output")
                output_ee = gr.HTML()

            with gr.Column(scale=1, visible=False) as compare_col:
                gr.Markdown("### Full Model (Comparison)")
                output_full = gr.HTML()

        status_html = gr.Markdown()

        with gr.Group():
            gr.Markdown("### Speedup Recap")
            speedup_md = gr.Markdown()
            with gr.Row():
                with gr.Column(visible=True) as ee_stats_col:
                    gr.Markdown("#### Early Exit")
                    ee_time = gr.Label(label="Time (s)")
                    ee_tps = gr.Label(label="Tokens/sec")
                    ee_avg = gr.Label(label="Avg Exit Layer")
                with gr.Column(visible=False) as full_stats_col:
                    gr.Markdown("#### Full Model")
                    full_time = gr.Label(label="Time (s)")
                    full_tps = gr.Label(label="Tokens/sec")
                    full_avg = gr.Label(label="Avg Exit Layer")

        def update_visibility(compare):
            return gr.update(visible=compare)

        compare_mode.change(
            fn=update_visibility,
            inputs=[compare_mode],
            outputs=[compare_col],
        )

        generate_btn.click(
            fn=generate,
            inputs=[
                prompt,
                model_selector,
                use_early_exit,
                accuracy_level,
                max_tokens,
                compare_mode,
            ],
            outputs=[
                output_ee,
                output_full,
                status_html,
                speedup_md,
                ee_time,
                ee_tps,
                ee_avg,
                full_time,
                full_tps,
                full_avg,
                ee_stats_col,
                full_stats_col,
                legend_html,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=False, debug=True)
