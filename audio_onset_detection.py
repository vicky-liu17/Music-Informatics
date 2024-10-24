import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import time
import warnings

# Suppress FutureWarning and UserWarnings from librosa
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Function to detect onsets in audio (using librosa)
def detect_onsets(file_path):
    """
    Detects the onsets in an audio file.

    Parameters:
    - file_path (str): Path to the audio file.

    Returns:
    - onset_times (np.ndarray): Detected onset times in seconds.
    """
    try:
        # Load the audio file using librosa
        y, sr = librosa.load(file_path, sr=None, mono=True)
        print(f"Loaded audio file '{file_path}' successfully with sample rate: {sr}")

        # Detect onsets (start of notes)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        print(f"Detected {len(onset_times)} onsets.")

        return onset_times

    except Exception as e:
        print(file_path)
        print(f"Error in detecting onsets: {str(e)}")
        return None


# Function to plot the waveform with detected onsets
def plot_onsets(file_path, onset_times):
    """
    Plots the waveform with detected onsets and saves the plot if it does not already exist.

    Parameters:
    - file_path (str): Path to the audio file.
    - onset_times (np.ndarray): Detected onset times in seconds.

    Returns:
    - output_plot (str): Path to the saved plot image.
    """
    try:
        # Get the base name of the file (without extension) and create a unique output plot name
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_plot = f"{base_name}_onsets_plot.png"

        # Check if the file already exists to avoid regenerating
        if os.path.exists(output_plot):
            print(f"Plot already exists: '{output_plot}', using existing image.")
            return output_plot

        # Load the audio file again for visualization
        y, sr = librosa.load(file_path, sr=None, mono=True)
        print(f"Loaded audio file '{file_path}' for plotting with sample rate: {sr}")

        # Plot the waveform
        plt.figure(figsize=(14, 6))
        librosa.display.waveshow(y, sr=sr, alpha=0.5)

        # Plot onsets (green)
        plt.vlines(onset_times, -1, 1, color='g', linestyle='--', label='Onsets')

        # Add labels and save the plot
        plt.title(f"Detected Onsets for {base_name}")
        plt.legend()
        plt.savefig(output_plot)
        plt.close()

        print(f"Plot saved as '{output_plot}'")
        return output_plot

    except Exception as e:
        print(f"Error in plotting onsets: {str(e)}")
        return None

# Example usage:
if __name__ == "__main__":
    # Example usage for testing
    audio_file = 'example_audio.wav'  # Replace with actual audio file
    onset_times = detect_onsets(audio_file)

    if onset_times is not None:
        plot_onsets(audio_file, onset_times)
