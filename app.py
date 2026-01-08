"""
DSSD Demo - Dynamic Self-Speculative Decoding Visualization
Showcases early exit inference with color-coded tokens showing which head generated each token.
"""

import gradio as gr
from pathlib import Path
from huggingface_hub import hf_hub_download

from src.inference import load_dssd_model, DSSDecoder, TokenInfo, StreamEvent

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
            f'border-radius: 3px; margin: 1px; display: inline-block;" title="{title}">{text}</span>'
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
            color = FULL_MODEL_COLOR
            title = "PENDING - Full Model"

        text = (
            token.token_text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        text = text.replace("\n", "<br>").replace(" ", "&nbsp;")

        html_parts.append(
            f'<span style="background-color: {color}; padding: 2px 4px; '
            f"border-radius: 3px; margin: 1px; display: inline-block; "
            f'border: 2px dashed #333; opacity: 0.7;" title="{title}">{text}</span>'
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


def create_stats_html(result, label: str) -> str:
    """Create statistics HTML display."""
    return f"""
    <div style="padding: 10px; background: #f5f5f5; border-radius: 8px; margin-top: 10px;">
        <h4 style="margin: 0 0 10px 0;">{label} Statistics</h4>
        <p><b>Time:</b> {result.total_time:.2f}s</p>
        <p><b>Tokens/sec:</b> {result.tokens_per_second:.2f}</p>
        <p><b>Avg Exit Layer:</b> {result.avg_exit_layer:.1f}</p>
        <p><b>Exit Distribution:</b> {result.exit_distribution}</p>
    </div>
    """


def generate(
    prompt: str,
    model_key: str,
    use_early_exit: bool,
    accuracy_level: float,
    max_tokens: int,
    compare_mode: bool,
):
    """Main generation function for Gradio interface with streaming."""
    try:
        decoder = get_decoder(model_key)
    except Exception as e:
        error_msg = f"<p style='color: red;'>Error loading model: {e}</p>"
        yield (error_msg, "", "", error_msg)
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
        for event in decoder.generate_streaming(
            prompt=prompt,
            max_tokens=int(max_tokens),
            accuracy_level=closest_level,
            use_chat_template=True,
        ):
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

            status = f"""
            <div style="padding: 10px; background: #fff3cd; border-radius: 8px;">
                <b>Early Exit:</b> {event.message} | <b>Full Model:</b> Waiting...
            </div>
            """

            yield (
                combined_html,
                "<p style='color: #666;'>Waiting for early exit to complete...</p>",
                status,
                legend,
            )
            final_ee_tokens = event.tokens

        # Now stream full model
        final_full_tokens = []
        for event in decoder.generate_full_model_streaming(
            prompt=prompt,
            max_tokens=int(max_tokens),
            use_chat_template=True,
        ):
            html_full = tokens_to_html(event.tokens, head_layers)
            status = f"""
            <div style="padding: 10px; background: #fff3cd; border-radius: 8px;">
                <b>Full Model:</b> {event.message}
            </div>
            """
            yield (
                tokens_to_html(final_ee_tokens, head_layers),
                html_full,
                status,
                legend,
            )
            final_full_tokens = event.tokens

        # Final stats
        result_ee = decoder.generate(
            prompt=prompt,
            max_tokens=int(max_tokens),
            use_early_exit=True,
            accuracy_level=closest_level,
            use_chat_template=True,
        )
        result_full = decoder.generate(
            prompt=prompt,
            max_tokens=int(max_tokens),
            use_early_exit=False,
            use_chat_template=True,
        )

        html_ee = tokens_to_html(result_ee.tokens, head_layers)
        html_full = tokens_to_html(result_full.tokens, head_layers)

        speedup = (
            result_ee.tokens_per_second / result_full.tokens_per_second
            if result_full.tokens_per_second > 0
            else 0
        )
        stats = f"""
        <div style="padding: 15px; background: #e8f5e9; border-radius: 8px;">
            <h3 style="margin: 0 0 10px 0;">🚀 Speedup: {speedup:.2f}x</h3>
            <div style="display: flex; gap: 20px;">
                <div style="flex: 1; padding: 10px; background: white; border-radius: 8px;">
                    <h4>Early Exit</h4>
                    <p><b>Time:</b> {result_ee.total_time:.2f}s | <b>Tokens/sec:</b> {result_ee.tokens_per_second:.2f}</p>
                    <p><b>Avg Exit Layer:</b> {result_ee.avg_exit_layer:.1f}</p>
                </div>
                <div style="flex: 1; padding: 10px; background: white; border-radius: 8px;">
                    <h4>Full Model</h4>
                    <p><b>Time:</b> {result_full.total_time:.2f}s | <b>Tokens/sec:</b> {result_full.tokens_per_second:.2f}</p>
                    <p><b>Avg Exit Layer:</b> {result_full.avg_exit_layer:.1f}</p>
                </div>
            </div>
        </div>
        """
        yield (html_ee, html_full, stats, legend)

    elif use_early_exit:
        # STREAMING mode for early exit - show draft/verify process
        for event in decoder.generate_streaming(
            prompt=prompt,
            max_tokens=int(max_tokens),
            accuracy_level=closest_level,
            use_chat_template=True,
        ):
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
            status = f"""
            <div style="padding: 10px; background: #fff3cd; border-radius: 8px; margin-top: 5px;">
                <b>Status:</b> {event.message}
            </div>
            """

            yield (combined_html, "", status, legend)

        # Final stats after streaming completes
        # Re-run to get final stats (or we could track during streaming)
        result = decoder.generate(
            prompt=prompt,
            max_tokens=int(max_tokens),
            use_early_exit=True,
            accuracy_level=closest_level,
            use_chat_template=True,
        )
        html = tokens_to_html(result.tokens, head_layers)
        stats = f"""
        <div style="padding: 15px; background: #f5f5f5; border-radius: 8px;">
            <h4 style="margin: 0 0 10px 0;">Early Exit Statistics (Final)</h4>
            <p><b>Tokens:</b> {len(result.tokens)} | <b>Tokens/sec:</b> {result.tokens_per_second:.2f} | <b>Avg Exit Layer:</b> {result.avg_exit_layer:.1f}</p>
            <p><b>Exit Distribution:</b> {result.exit_distribution}</p>
        </div>
        """
        yield (html, "", stats, legend)

    else:
        # Full model mode (streaming)
        for event in decoder.generate_full_model_streaming(
            prompt=prompt,
            max_tokens=int(max_tokens),
            use_chat_template=True,
        ):
            html = tokens_to_html(event.tokens, head_layers)
            status = f"""
            <div style="padding: 10px; background: #fff3cd; border-radius: 8px;">
                <b>Full Model:</b> {event.message}
            </div>
            """
            yield (html, "", status, legend)

        # Final stats
        result = decoder.generate(
            prompt=prompt,
            max_tokens=int(max_tokens),
            use_early_exit=False,
            use_chat_template=True,
        )
        html = tokens_to_html(result.tokens, head_layers)
        stats = f"""
        <div style="padding: 15px; background: #f5f5f5; border-radius: 8px;">
            <h4 style="margin: 0 0 10px 0;">Full Model Statistics</h4>
            <p><b>Tokens:</b> {len(result.tokens)} | <b>Time:</b> {result.total_time:.2f}s | <b>Tokens/sec:</b> {result.tokens_per_second:.2f}</p>
        </div>
        """
        yield (html, "", stats, legend)


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

        # Stats (full width)
        stats_html = gr.HTML()

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
            outputs=[output_ee, output_full, stats_html, legend_html],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=False)
