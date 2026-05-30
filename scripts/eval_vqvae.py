import os
import sys
import json
import torch
from pathlib import Path
from transformers import PreTrainedTokenizerFast

# Add parent dir to sys.path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.vqvae.model import SemanticTextVQVAE

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    with open("configs/vqvae_train.json") as f:
        cfg = json.load(f)
        
    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer/tokenizer.json")
    tokenizer.pad_token = "<|pad|>"
    tokenizer.bos_token = "<|endoftext|>"
    tokenizer.eos_token = "<|endoftext|>"
    
    print("Loading model...")
    # Load model
    model = SemanticTextVQVAE(
        vocab_size=cfg.get("vocab_size", 32000),
        hidden_size=cfg.get("hidden_size", 1024),
        num_semantic_tokens=cfg.get("num_semantic_tokens", 4096),
        semantic_sequence_length=cfg.get("semantic_sequence_length", 128),
        pad_token_id=cfg.get("pad_token_id", 0),
        semantic_pad_token_id=cfg.get("semantic_pad_token_id", 0),
        max_position_embeddings=cfg.get("max_position_embeddings", 2048),
        use_rotary_embeddings=cfg.get("use_rotary_embeddings", True)
    ).to(device)
    
    # Load checkpoint
    ckpt_path = "checkpoints/vqvae.pt"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        return
        
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Remove `_orig_mod.` prefix added by torch.compile
    state_dict = {}
    for k, v in ckpt["model"].items():
        if k.startswith("_orig_mod."):
            state_dict[k[len("_orig_mod."):]] = v
        else:
            state_dict[k] = v
            
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.\n")
    
    test_texts = [
        "def hello_world():\n    print('Hello, world!')",
        "import torch\nimport torch.nn as nn\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(10, 2)\n\n    def forward(self, x):\n        return self.fc(x)"
    ]
    
    print("Testing VQ-VAE model reconstruction...\n")
    
    with torch.no_grad():
        for i, text in enumerate(test_texts):
            print(f"--- Example {i+1} ---")
            print("ORIGINAL:")
            print(text)
            print("-" * 40)
            
            # Encode text
            tokens = tokenizer.encode(text, return_tensors="pt").to(device)
            
            # Encode to semantic indices
            semantic_indices, _ = model.encode_sentence(tokens)
            
            bos_id = int(tokens[0, 0].item())
            
            # Decode back
            decoded_tokens = model.decode_from_semantic_indices(
                semantic_indices,
                max_length=tokens.size(1) + 20,
                bos_token_id=bos_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=0.1,  # low temperature for stable greedy-like reconstruction
            )
            
            # Print result
            decoded_text = tokenizer.decode(decoded_tokens[0], skip_special_tokens=True)
            print("RECONSTRUCTED:")
            print(decoded_text)
            print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
