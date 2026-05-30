import argparse
import torch
import math
from pathlib import Path

def migrate_checkpoint(input_path: str, output_path: str, new_fsq_dim: int = 6):
    print(f"Loading {input_path}...")
    ckpt = torch.load(input_path, map_location="cpu")
    
    # Check if it's a full payload or just state_dict
    state_dict = ckpt.get("model", ckpt)
    is_payload = "model" in ckpt
    
    new_state_dict = {}
    
    # Track which keys we've mapped
    mapped_keys = set()
    
    for k, v in state_dict.items():
        # 1. Remove pos_embedding
        if "pos_embedding" in k:
            print(f"Removing {k}")
            mapped_keys.add(k)
            continue
            
        # 2. Map norm1 to norm_self_attn, norm_cross_attn, norm_ffn in SDPADecoderLayer
        if ".norm1." in k:
            print(f"Mapping {k} to sequential norms")
            new_k_self = k.replace(".norm1.", ".norm_self_attn.")
            new_k_cross = k.replace(".norm1.", ".norm_cross_attn.")
            new_k_ffn = k.replace(".norm1.", ".norm_ffn.")
            
            new_state_dict[new_k_self] = v.clone()
            new_state_dict[new_k_cross] = v.clone()
            new_state_dict[new_k_ffn] = v.clone()
            mapped_keys.add(k)
            continue
            
        # 3. Handle pre_quant_proj shape mismatch if fsq_dim changed
        if "pre_quant_proj." in k:
            old_dim = v.size(0)
            if old_dim != new_fsq_dim:
                print(f"Resizing {k} from {old_dim} to {new_fsq_dim}")
                if "weight" in k:
                    new_v = torch.randn(new_fsq_dim, v.size(1)) * 0.02
                    min_dim = min(old_dim, new_fsq_dim)
                    new_v[:min_dim] = v[:min_dim]
                elif "bias" in k:
                    new_v = torch.zeros(new_fsq_dim)
                    min_dim = min(old_dim, new_fsq_dim)
                    new_v[:min_dim] = v[:min_dim]
                new_state_dict[k] = new_v
                mapped_keys.add(k)
                continue
                
        # 4. Handle post_quant_proj shape mismatch if fsq_dim changed
        if "post_quant_proj.weight" in k:
            old_dim = v.size(1)
            if old_dim != new_fsq_dim:
                print(f"Resizing {k} from {old_dim} to {new_fsq_dim}")
                new_v = torch.randn(v.size(0), new_fsq_dim) * 0.02
                min_dim = min(old_dim, new_fsq_dim)
                new_v[:, :min_dim] = v[:, :min_dim]
                new_state_dict[k] = new_v
                mapped_keys.add(k)
                continue
                
        # Pass through unchanged keys
        if k not in mapped_keys:
            new_state_dict[k] = v
            
    # Add new parameters
    print("Adding mask_token...")
    hidden_size = state_dict["embedding.weight"].size(1)
    new_state_dict["mask_token"] = torch.randn(1, 1, hidden_size) * 0.02
    
    print("Adding downsample.input_norm weights...")
    # Get downsample dimension
    downsample_dim = hidden_size
    new_state_dict["downsample.input_norm.weight"] = torch.ones(downsample_dim)
    new_state_dict["downsample.input_norm.bias"] = torch.zeros(downsample_dim)
    
    print("Adding residual_scale to encoder layers...")
    # Find number of encoder layers
    encoder_layers = set()
    for k in state_dict.keys():
        if k.startswith("encoder.layers."):
            layer_idx = k.split(".")[2]
            encoder_layers.add(layer_idx)
    
    total_layers = len(encoder_layers)
    if total_layers > 0:
        alpha = (2.0 * max(1, total_layers)) ** 0.25
        for idx in encoder_layers:
            new_state_dict[f"encoder.layers.{idx}.residual_scale"] = torch.ones(hidden_size) / alpha
    
    # Save the updated dict
    if is_payload:
        ckpt["model"] = new_state_dict
        torch.save(ckpt, output_path)
    else:
        torch.save(new_state_dict, output_path)
        
    print(f"Successfully migrated checkpoint to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the old checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Path to save the migrated checkpoint")
    parser.add_argument("--fsq_dim", type=int, default=6, help="New FSQ dimension (length of fsq_levels)")
    args = parser.parse_args()
    
    migrate_checkpoint(args.input, args.output, args.fsq_dim)
