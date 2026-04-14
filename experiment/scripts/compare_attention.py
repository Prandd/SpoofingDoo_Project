import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import soundfile as sf
import torch

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.aasist_imported import Model

def pad(x, max_len=64600):
    """Pads or truncates audio to the exact length AASIST expects (~4 seconds)"""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x

def extract_attention(model, device, audio_path, target_layer="GAT_layer_S"):
    """Runs audio through the model and extracts attention from one chosen layer."""
    if not os.path.exists(audio_path):
        print(f"Error: Could not find {audio_path}")
        return None
        
    # --- FIX 1: Wipe the model's memory of any old attention maps ---
    for module in model.modules():
        if hasattr(module, 'saved_att_map'):
            delattr(module, 'saved_att_map')
            
    X, sr = sf.read(audio_path)
    if X.ndim > 1:
        # Convert stereo/multi-channel audio to mono so model input shape is consistent.
        X = np.mean(X, axis=1)
    X_pad = pad(X)
    
    # Add batch dimension and send to device
    x_inp = torch.Tensor(X_pad).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, _ = model(x_inp)
        
    attention_matrix = None
    for name, module in model.named_modules():
        if name == target_layer and hasattr(module, 'saved_att_map'):
            # Use one explicit layer to avoid accidentally picking a wrong/stale map.
            attention_matrix = module.saved_att_map[0].detach().cpu().numpy().copy()
            print(f"  -> Successfully extracted fresh graph from layer: {name}")
            break
            
    if attention_matrix is None:
        print(
            f"  -> Error: No saved_att_map found for {audio_path}! "
            f"Layer '{target_layer}' was not executed."
        )
        return None
        
    # Remove the extra dimension
    attention_matrix = np.squeeze(attention_matrix)
    
    return attention_matrix

def build_model(config, device, state_dict=None):
    """Create a fresh model instance."""
    model = Model(config["model_config"]).to(device)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.eval()
    return model

def extract_genuine_attention(config, device, model_state_dict, genuine_audio_path, target_layer="GAT_layer_S"):
    """Extract attention for genuine audio using its own model instance."""
    model_genuine = build_model(config, device, state_dict=model_state_dict)
    return extract_attention(model_genuine, device, genuine_audio_path, target_layer=target_layer)

def extract_mms_spoof_attention(config, device, model_state_dict, spoof_audio_path, target_layer="GAT_layer_S"):
    """Extract attention for spoof audio using its own model instance."""
    model_spoof = build_model(config, device, state_dict=model_state_dict)
    return extract_attention(model_spoof, device, spoof_audio_path, target_layer=target_layer)

if __name__ == "__main__":
    # 1. Load the Configuration
    config_path = "../aasist-main/aasist-main/config/AASIST.conf" 
    
    try:
        with open(config_path, "r") as f:
            config = json.loads(f.read())
    except FileNotFoundError:
        print(f"Config file not found at {config_path}. Please verify the path.")
        exit()

    # 2. Choose device
    device = torch.device("cpu") 
    torch.manual_seed(42)
    np.random.seed(42)
    # Build one base model, then reuse exactly the same weights for both extractions.
    base_model = build_model(config, device)
    base_model_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}

    # 3. Define your audio pairs
    _root = Path(__file__).resolve().parents[2]
    genuine_audio = str(_root / "data" / "raw" / "wav" / "genuine" / "G1" / "thai0001.wav")
    mms_spoof_audio = str(_root / "data" / "raw" / "wav" / "MMS_spoof" / "mms_1" / "mms_thai0001.wav")
    if not Path(genuine_audio).exists():
        genuine_audio = str(_root / "extract_audio" / "genuine" / "G1" / "thai0001.wav")
    if not Path(mms_spoof_audio).exists():
        mms_spoof_audio = str(_root / "extract_audio" / "MMS_spoof" / "mms_1" / "mms_thai0001.wav")

    print("Extracting Genuine Graph...")
    genuine_attention = extract_genuine_attention(
        config, device, base_model_state, genuine_audio, target_layer="GAT_layer_S"
    )
    
    print("\nExtracting MMS Spoof Graph...")
    mms_spoof_attention = extract_mms_spoof_attention(
        config, device, base_model_state, mms_spoof_audio, target_layer="GAT_layer_S"
    )

    # 4. Plot them side-by-side
    if genuine_attention is not None and mms_spoof_attention is not None:
        print(
            f"Sanity check (mean values): genuine={genuine_attention.mean():.6f}, "
            f"mms_spoof={mms_spoof_attention.mean():.6f}"
        )
        plt.figure(figsize=(16, 7))

        # Plot Genuine
        plt.subplot(1, 2, 1)
        sns.heatmap(genuine_attention.squeeze(), cmap='magma', cbar=True)
        plt.title('Genuine Audio: Attention Graph')

        # Plot MMS Spoof
        plt.subplot(1, 2, 2)
        sns.heatmap(mms_spoof_attention.squeeze(), cmap='magma', cbar=True)
        plt.title('MMS Spoof Audio: Attention Graph')

        plt.tight_layout()
        plt.show()