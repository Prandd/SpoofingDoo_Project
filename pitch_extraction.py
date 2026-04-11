import librosa
import numpy as np
import matplotlib.pyplot as plt

def pad_or_truncate(x, max_len=64600):
    """Ensures audio is exactly the length AASIST expects."""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x

def extract_and_normalize_pitch(audio_path, sr=16000, max_len=64600):
    """
    Extracts the F0 (pitch) contour from an audio file and prepares it for a Neural Network.
    """
    # 1. Load and format the audio
    y, _ = librosa.load(audio_path, sr=sr)
    y = pad_or_truncate(y, max_len)
    
    # 2. Extract Pitch using pYIN
    # fmin and fmax represent the normal range of human speech (approx 65Hz to 1046Hz)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C6'),
        sr=sr,
        frame_length=2048,
        hop_length=512 # This dictates the temporal resolution of the output
    )
    
    # 3. Clean the Data (Crucial for Neural Networks)
    # pYIN returns 'NaN' (Not a Number) for unvoiced frames. 
    # Neural networks will crash if fed NaNs, so we convert them to 0.0
    f0_clean = np.nan_to_num(f0, nan=0.0)
    
    # 4. Normalize the Pitch
    # Neural networks prefer inputs roughly between -1 and 1.
    # We use Z-score normalization (subtract mean, divide by standard deviation)
    # We only calculate the mean/std based on VOICED frames to not skew the data.
    voiced_pitch = f0_clean[f0_clean > 0]
    
    if len(voiced_pitch) > 0:
        mean_pitch = np.mean(voiced_pitch)
        std_pitch = np.std(voiced_pitch)
        f0_normalized = (f0_clean - mean_pitch) / (std_pitch + 1e-8)
        # Ensure unvoiced frames remain exactly 0 after normalization
        f0_normalized = np.where(f0_clean == 0.0, 0.0, f0_normalized)
    else:
        # Failsafe if the audio is complete silence
        f0_normalized = f0_clean
        
    return f0_normalized

# ==========================================
# Testing the Extraction
# ==========================================
if __name__ == "__main__":
    genuine_audio = "./extract_audio/CSS/Bona fide/Formal/Formal_spk1_utt1.wav"
    mms_spoof_audio = "./extract_audio/CSS/Spoofed/Formal/VITS_Formal_spk1_utt1.wav"
    
    # Extract pitch
    gen_pitch = extract_and_normalize_pitch(genuine_audio)
    spoof_pitch = extract_and_normalize_pitch(mms_spoof_audio)
    
    print(f"Original Audio Length (samples): 64600")
    print(f"Extracted Pitch Array Length: {len(gen_pitch)}")
    
    # Plot the results to visually compare the Thai tonal fluidity
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(gen_pitch, color='blue')
    plt.title("Normalized Pitch Contour: Genuine Thai Audio")
    plt.ylabel("Normalized F0")
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(spoof_pitch, color='red')
    plt.title("Normalized Pitch Contour: MMS Spoof Audio")
    plt.ylabel("Normalized F0")
    plt.xlabel("Time Frames")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()