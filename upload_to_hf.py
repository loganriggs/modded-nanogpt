"""
Script to upload a trained model checkpoint to Hugging Face Hub.

Usage:
    python upload_to_hf.py --checkpoint logs/run_name/state_step000000.pt --repo_id username/model-name
"""

import argparse
import torch
from huggingface_hub import HfApi, create_repo
import json
from dataclasses import asdict
import sys

# Import the model classes from train_gpt2
from train_gpt2 import GPT, GPTConfig

def upload_model_to_hf(checkpoint_path, repo_id, token=None, private=False):
    """
    Upload a model checkpoint to Hugging Face Hub.

    Args:
        checkpoint_path: Path to the checkpoint file (.pt)
        repo_id: Repository ID on HF Hub (e.g., "username/model-name")
        token: HF token (optional, uses cached token if not provided)
        private: Whether to make the repo private
    """

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract model config and state dict
    config = checkpoint['config']
    state_dict = checkpoint['model']
    step = checkpoint.get('step', 'unknown')

    print(f"Model config: {config}")
    print(f"Training step: {step}")

    # Remove _orig_mod. prefix from state_dict keys (from torch.compile)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # Create the model
    model = GPT(config)
    model.load_state_dict(state_dict)

    # Create repository
    api = HfApi(token=token)
    try:
        create_repo(repo_id, private=private, token=token, exist_ok=True)
        print(f"Created/verified repository: {repo_id}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        sys.exit(1)

    # Save model locally first
    temp_dir = "temp_hf_upload"
    import os
    os.makedirs(temp_dir, exist_ok=True)

    # Save model weights
    model_path = os.path.join(temp_dir, "pytorch_model.bin")
    torch.save(state_dict, model_path)
    print(f"Saved model weights to {model_path}")

    # Save config as JSON
    config_dict = asdict(config)
    config_dict['step'] = step
    config_path = os.path.join(temp_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved config to {config_path}")

    # Create a simple README
    readme_path = os.path.join(temp_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(f"""# Modded NanoGPT Model

This is a GPT-2 style model trained with modifications from modded-nanogpt.

## Model Config

- Layers: {config.n_layer}
- Heads: {config.n_head}
- Embedding dimension: {config.n_embd}
- Vocab size: {config.vocab_size}
- Squared MLP: {config.squared_mlp}
- Bilinear: {config.bilinear}
- Gated: {config.gated}
- Expansion factor: {config.expansion_factor}

## Training

- Training step: {step}

## Usage

```python
from huggingface_hub import hf_hub_download
import torch
from train_gpt2 import GPT, GPTConfig
import json

# Download config
config_path = hf_hub_download(repo_id="{repo_id}", filename="config.json")
with open(config_path) as f:
    config_dict = json.load(f)

# Remove non-GPTConfig fields
config_dict.pop('step', None)

# Create model
config = GPTConfig(**config_dict)
model = GPT(config)

# Download and load weights
weights_path = hf_hub_download(repo_id="{repo_id}", filename="pytorch_model.bin")
state_dict = torch.load(weights_path, map_location='cpu')
model.load_state_dict(state_dict)

model.eval()
```
""")
    print(f"Created README at {readme_path}")

    # Upload files
    print("Uploading to Hugging Face Hub...")
    api.upload_folder(
        folder_path=temp_dir,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )

    print(f"✓ Model successfully uploaded to https://huggingface.co/{repo_id}")

    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_dir)
    print("Cleaned up temporary files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload model checkpoint to Hugging Face Hub")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--repo_id", type=str, required=True, help="HF repository ID (username/model-name)")
    parser.add_argument("--token", type=str, default=None, help="HF token (optional)")
    parser.add_argument("--private", action="store_true", help="Make repository private")

    args = parser.parse_args()

    upload_model_to_hf(args.checkpoint, args.repo_id, args.token, args.private)
