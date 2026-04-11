import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Import the model architecture directly from your local AASIST_imported.py
from AASIST_imported import Model 

def pad(x, max_len=64600):
    """Pads or truncates audio to the exact length AASIST expects (~4 seconds)"""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x

def build_model(config, device, weights_path):
    """Creates the model and STRICTLY loads trained weights."""
    model = Model(config["model_config"]).to(device)
    
    print(f"Attempting to load weights from: {weights_path}")
    # STRICT LOAD: No try/except. It must crash here if the path is wrong!
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print("-> Weights loaded successfully!")
    
    model.eval()
    return model

def extract_attention(model, device, audio_path, target_layer="GAT_layer_S"):
    """Runs audio through the model and extracts attention from one chosen layer."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Error: Could not find audio at {audio_path}")
        
    # Wipe the model's memory of any old attention maps
    for module in model.modules():
        if hasattr(module, 'saved_att_map'):
            delattr(module, 'saved_att_map')
            
    X, sr = sf.read(audio_path)
    if X.ndim > 1:
        # Convert stereo/multi-channel audio to mono so model input shape is consistent
        X = np.mean(X, axis=1)
    X_pad = pad(X)
    
    # Add batch dimension and send to device
    x_inp = torch.Tensor(X_pad).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, _ = model(x_inp)
        
    attention_matrix = None
    for name, module in model.named_modules():
        if name == target_layer and hasattr(module, 'saved_att_map'):
            attention_matrix = module.saved_att_map[0].detach().cpu().numpy().copy()
            print(f"  -> Successfully extracted fresh graph from layer: {name}")
            break
            
    if attention_matrix is None:
        raise ValueError(
            f"  -> Error: No saved_att_map found! "
            f"Layer '{target_layer}' was not executed or is missing the interception line."
        )
        
    # Remove the extra dimension so seaborn can plot it
    attention_matrix = np.squeeze(attention_matrix)
    
    return attention_matrix

if __name__ == "__main__":
    # ==========================================
    # 1. SETUP PATHS - VERIFY THESE ARE CORRECT
    # ==========================================
    config_path = "../aasist-main/aasist-main/config/AASIST.conf" 
    
    # ---> PASTE YOUR EXACT WEIGHTS PATH HERE <---
    weights_path = "../aasist-main/aasist-main/models/weights/AASIST.pth" 
    
    # Target audio file
    audio_path = "./extract_audio/CSS/Spoofed/Formal/VITS_Formal_spk1_utt1.wav"
    
    # The layer you want to intercept
    target_layer = "GAT_layer_S" 
    
    # ==========================================
    # 2. LOAD CONFIG & MODEL
    # ==========================================
    try:
        with open(config_path, "r") as f:
            config = json.loads(f.read())
    except FileNotFoundError:
        print(f"Config file not found at {config_path}. Please verify the path.")
        exit()

    device = torch.device("cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize model with weights
    model = build_model(config, device, weights_path)

    # ==========================================
    # 3. EXTRACT AND PLOT
    # ==========================================
    print(f"\nExtracting Graph for: {audio_path}")
    attention_map = extract_attention(model, device, audio_path, target_layer=target_layer)

    print(f"Sanity check (mean value): {attention_map.mean():.6f}")

    # Plot the heatmap
    plt.figure(figsize=(9, 7))
    sns.heatmap(attention_map, cmap='magma', cbar=True)
    plt.title(f'Attention Graph ({target_layer})\nFile: {os.path.basename(audio_path)}')
    plt.tight_layout()
    plt.show()