import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def extract_and_compare_features(genuine_path, spoofed_path):
    """
    Loads a pair of audio files, extracts their Mel-spectrograms (spectro-temporal features),
    and plots them side-by-side for comparative analysis.
    """
    gen_file = Path(genuine_path)
    spoof_file = Path(spoofed_path)

    if not gen_file.exists() or not spoof_file.exists():
        print("Error: One or both of the audio files could not be found.")
        return

    print(f"Analyzing Pair:\nGenuine: {gen_file.name}\nSpoofed: {spoof_file.name}\n")

    # 1. Load the audio files
    # sr=None preserves the original sample rate
    y_gen, sr_gen = librosa.load(gen_file, sr=None)
    y_spoof, sr_spoof = librosa.load(spoof_file, sr=None)

    # 2. Extract Spectro-Temporal Features (Mel-Spectrograms)
    # n_fft is the window size, hop_length is the step size between windows
    n_fft = 2048
    hop_length = 512

    # Calculate Mel-spectrogram for Genuine
    S_gen = librosa.feature.melspectrogram(y=y_gen, sr=sr_gen, n_fft=n_fft, hop_length=hop_length)
    S_gen_db = librosa.power_to_db(S_gen, ref=np.max)

    # Calculate Mel-spectrogram for Spoofed
    S_spoof = librosa.feature.melspectrogram(y=y_spoof, sr=sr_spoof, n_fft=n_fft, hop_length=hop_length)
    S_spoof_db = librosa.power_to_db(S_spoof, ref=np.max)

    # 3. Create the comparative graph/plot
    plt.figure(figsize=(14, 6))

    # Plot Genuine Graph
    plt.subplot(1, 2, 1)
    librosa.display.specshow(S_gen_db, sr=sr_gen, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.title('Genuine Spectro-Temporal Graph')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    # Plot Spoofed Graph
    plt.subplot(1, 2, 2)
    librosa.display.specshow(S_spoof_db, sr=sr_spoof, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.title(f'Spoofed Spectro-Temporal Graph')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Point these to your exact extracted file locations
    base_dir = "./extract_audio"
    
    genuine_audio = f"{base_dir}/CSS/Bona fide/Formal/Formal_spk1_utt1.wav"
    spoofed_audio = f"{base_dir}/CSS/Spoofed/Formal/VITS_Formal_spk1_utt1.wav"
    
    extract_and_compare_features(genuine_audio, spoofed_audio)