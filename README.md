# DSSD Demo - Dynamic Self-Speculative Decoding

A Gradio demo showcasing early exit inference with color-coded token visualization.

## Features

- **Color-coded tokens**: Each token shows which head/layer generated it
- **True early exit**: Actual speedup by stopping layer computation early
- **Compare mode**: Side-by-side comparison with full model
- **Model selection**: Switch between different DSSD models

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo
python app.py
```

Then open http://localhost:7860 in your browser.

## Models

- **DSSD-Llama3-8B**: Llama 3 8B with 3 early exit heads at layers 8, 16, 24
- **DSSD-Qwen3-0.6B**: Qwen3 0.6B with 4 early exit heads at layers 5, 11, 16, 22

## Color Legend

- 🔴 **Red**: Head 0 (earliest layer)
- 🟠 **Orange**: Head 1
- 🔵 **Teal/Blue**: Head 2-3
- 🟢 **Light Green**: Full model (all layers)
