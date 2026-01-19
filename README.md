---
title: DSSD Demo
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.3.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 🚀 Dynamic Self-Speculative Decoding (DSSD) Demo

This demo showcases **early exit inference** with true speculative decoding. 
Tokens are generated from intermediate layers when the model is confident, resulting in faster generation while **guaranteeing output identical to the full model**.

## Features
- **Speculative Decoding**: Uses early exit heads to draft tokens, then verifies them with the full model.
- **Streaming Output**: Watch the generation process live, including drafting and verification statuses.
- **Model Comparison**: Compare performance and output between DSSD and the full model side-by-side.
- **Color-coded Visualization**: Each token is colored based on which head/layer generated it.

## How it works
1. **Draft Phase**: The model tries to predict the next token(s) using early exit heads placed at intermediate layers.
2. **Verification Phase**: The full model checks the drafted tokens in a single forward pass.
3. **Acceptance**: Matching tokens are kept. The first mismatch is corrected, and the process restarts.

## Models
- **Llama 3 8B**: Using 3 auxiliary heads at layers 8, 16, and 24.
- **Qwen 3 0.6B**: Using 4 auxiliary heads at layers 5, 11, 16, and 22.

## Quick Start (Local)

```bash
pip install -r requirements.txt
python app.py
```