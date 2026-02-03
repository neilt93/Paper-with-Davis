#!/bin/bash
# RunPod Setup Script for VLM Visibility Evaluation
# Run this on a RunPod instance with RTX 3090/4090 (24GB) or A100 (40/80GB)

set -e

echo "=== Setting CUDA memory config (prevents fragmentation OOM) ==="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Pin transformers to stable version (5.x has breaking API changes)
# See: https://github.com/huggingface/transformers/issues/40242
pip install "transformers>=4.45.0,<5.0.0" accelerate pillow pandas python-dotenv

pip install qwen-vl-utils  # For Qwen models
pip install timm einops  # For InternVL2
pip install bitsandbytes  # For 4-bit quantization (optional)

# Optional: flash-attn for faster InternVL2 (requires compilation, may take a while)
# pip install flash-attn --no-build-isolation

# Optional: if models are gated (Llama, Molmo, Gemma)
# export HF_TOKEN="your_token_here"
# huggingface-cli login --token $HF_TOKEN

echo "=== Verifying GPU ==="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

echo "=== Pre-downloading models (optional, run if needed) ==="
echo "To pre-download a model before running inference:"
echo "  python -c \"from transformers import AutoModel; AutoModel.from_pretrained('OpenGVLab/InternVL2-8B', trust_remote_code=True)\""

echo "=== Patching InternVL2 (if downloaded) ==="
python scripts/patch_internvl.py || echo "InternVL2 patch skipped (model not yet downloaded)"

echo "=== Setup complete ==="
echo ""
echo "IMPORTANT: Add this to your shell before running models:"
echo "  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo ""
echo "Recommended VLLMs for RunPod (tested working):"
echo ""
echo "  # InternVL3-8B (RECOMMENDED - native HF support, no patch needed)"
echo "  python scripts/local/run_visibility_eval.py --input Sheets/FINAL_Pictures_DB.csv --model internvl3-8b"
echo ""
echo "  # Phi-3.5-Vision (fast, lightweight, ~8GB VRAM)"
echo "  python scripts/local/run_visibility_eval.py --input Sheets/FINAL_Pictures_DB.csv --model phi3-vision"
echo ""
echo "  # SmolVLM (smallest, fastest, ~4GB VRAM)"
echo "  python scripts/local/run_visibility_eval.py --input Sheets/FINAL_Pictures_DB.csv --model smolvlm"
echo ""
echo "  # Qwen2.5-VL-7B (strong quality, ~16GB VRAM)"
echo "  python scripts/local/run_visibility_eval.py --input Sheets/FINAL_Pictures_DB.csv --model qwen-vl-7b"
echo ""
echo "  # LLaVA-OneVision-7B (~16GB VRAM)"
echo "  python scripts/local/run_visibility_eval.py --input Sheets/FINAL_Pictures_DB.csv --model llava-7b"
echo ""
echo "  # InternVL2-8B (legacy, requires patch after first download)"
echo "  python scripts/local/run_visibility_eval.py --input Sheets/FINAL_Pictures_DB.csv --model internvl2-8b"
echo "  # Then run: python scripts/patch_internvl.py"
echo ""
echo "If you get OOM errors, try: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
