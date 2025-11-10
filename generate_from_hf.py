"""
Script to load a model from Hugging Face Hub and generate text.

Usage:
    python generate_from_hf.py --repo_id username/model-name --prompt "Once upon a time"
"""

import argparse
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
import json

# Import model classes
from train_gpt2 import GPT, GPTConfig

def load_model_from_hf(repo_id, device='cuda'):
    """
    Load a model from Hugging Face Hub.

    Args:
        repo_id: Repository ID on HF Hub (e.g., "username/model-name")
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        model: Loaded GPT model
        config: Model configuration
    """
    print(f"Downloading model from {repo_id}...")

    # Download config
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    # Remove non-GPTConfig fields
    step = config_dict.pop('step', None)
    print(f"Model was saved at training step: {step}")

    # Create model
    config = GPTConfig(**config_dict)
    print(f"Model config: {config}")

    model = GPT(config)

    # Download and load weights
    weights_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    return model, config

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Generate text from a prompt.

    Args:
        model: GPT model
        idx: Tensor of shape (B, T) with prompt tokens
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: If set, only sample from top k tokens

    Returns:
        Tensor of shape (B, T + max_new_tokens) with generated tokens
    """
    for _ in range(max_new_tokens):
        # Crop context if needed (model has max context length based on training)
        # For simplicity, let's use the last 1024 tokens
        idx_cond = idx if idx.size(1) <= 1024 else idx[:, -1024:]

        # Forward pass - we need to modify this since our model expects targets too
        # Let's create dummy targets for generation
        with torch.no_grad():
            x = model.transformer.wte(idx_cond)
            x = F.rms_norm(x, (x.size(-1),))
            x0 = x
            v1 = None
            for block in model.transformer.h:
                x, v1 = block(x, v1, x0)
            x = F.rms_norm(x, (x.size(-1),))
            logits = model.lm_head(x)
            logits = 30 * torch.tanh(logits / 30)

        # Take logits at the last position
        logits = logits[:, -1, :] / temperature

        # Optionally crop to top k
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def main():
    parser = argparse.ArgumentParser(description="Generate text from HF model")
    parser.add_argument("--repo_id", type=str, required=True, help="HF repository ID")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=200, help="Top-k sampling")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Load model
    model, config = load_model_from_hf(args.repo_id, args.device)

    # Load tokenizer (using GPT-2 tokenizer as that's what the model was trained with)
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        print("Loaded GPT-2 tokenizer")
    except ImportError:
        print("transformers library not found. Install with: pip install transformers")
        return

    # Encode prompt
    prompt_tokens = tokenizer.encode(args.prompt)
    print(f"\nPrompt: {args.prompt}")
    print(f"Prompt tokens: {prompt_tokens}")

    # Generate
    print(f"\nGenerating {args.max_tokens} tokens...")
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=args.device)

    generated_idx = generate(
        model,
        idx,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )

    # Decode
    generated_tokens = generated_idx[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)

    print("\n" + "="*80)
    print("GENERATED TEXT:")
    print("="*80)
    print(generated_text)
    print("="*80)

if __name__ == "__main__":
    main()
