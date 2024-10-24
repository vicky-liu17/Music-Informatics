import librosa
import numpy as np
import noisereduce as nr
from scipy.signal import butter, lfilter
import os
import soundfile as sf  # Use soundfile instead of librosa.output

def preprocess_audio(file_path, target_db=-20):
    """
    Preprocesses the audio file by normalizing the volume, reducing noise, and applying a high-pass filter.

    Parameters:
    - file_path (str): Path to the audio file.
    - target_db (float): Target decibel level for normalization.

    Returns:
    - output_file (str): Path to the saved preprocessed audio file.
    """

    # Load the audio file
    y, sr = librosa.load(file_path, sr=None, mono=True)

    # 1. Normalize the volume to a target decibel level
    # y = normalize_volume(y, target_db)

    # 2. Reduce noise using noisereduce library
    y = nr.reduce_noise(y=y, sr=sr)

    # 3. Apply a high-pass filter to remove low-frequency noise
    # y = apply_highpass_filter(y, sr, cutoff=100)

    # Generate the output file path based on input file name
    base_name, ext = os.path.splitext(os.path.basename(file_path))
    output_file = f"{base_name}_preprocessed.wav"
    
    # Save the preprocessed audio using soundfile instead of librosa.output
    sf.write(output_file, y, sr)

    return output_file  # Return the path to the preprocessed file

def normalize_volume(y, target_db=-20):
    """
    Normalize the audio volume to the target decibel level.

    Parameters:
    - y (np.ndarray): Audio signal.
    - target_db (float): Target decibel level for normalization.

    Returns:
    - y (np.ndarray): Normalized audio signal.
    """
    rms = np.sqrt(np.mean(y**2))
    scalar = 10**(target_db / 20) / (rms + 1e-6)
    return y * scalar

def apply_highpass_filter(y, sr, cutoff=100, order=5):
    """
    Apply a high-pass filter to remove low-frequency noise.

    Parameters:
    - y (np.ndarray): Audio signal.
    - sr (int): Sample rate of the audio signal.
    - cutoff (float): Cutoff frequency for the high-pass filter.
    - order (int): Filter order.

    Returns:
    - y (np.ndarray): Filtered audio signal.
    """
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, y)

# Example usage for testing:
if __name__ == "__main__":
    audio_file = 'practice_audio.wav'  # Replace with your actual practice file
    preprocessed_file = preprocess_audio(audio_file)
    print(f"Preprocessed audio saved as: {preprocessed_file}")
