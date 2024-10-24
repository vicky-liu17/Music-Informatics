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

        # Normalize the waveform to range [-1, 1]
        y = y / np.max(np.abs(y))

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

def extract_onset_features(file_path, sr=22050):
    """
    Extract onset times and onset strength from the audio file and align them.
    
    :param file_path: Path to the audio file
    :param sr: Sampling rate, default is 22050Hz
    :return: onset times, aligned and normalized onset strength
    """
    # Load the audio file
    y, sr = librosa.load(file_path, sr=sr)
    
    # Extract onset strength envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    # Extract onset times (in frames, then convert to time)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    # Get time corresponding to the strength envelope
    strength_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)
    
    # Align onset strength with the onset times by picking the nearest strength for each onset time
    aligned_onset_strength = np.interp(onset_times, strength_times, onset_env)
    
    # Normalize the aligned onset strength to the range [0, 1]
    max_strength = np.max(aligned_onset_strength)
    if max_strength > 0:
        aligned_onset_strength = aligned_onset_strength / max_strength
    
    return onset_times, aligned_onset_strength

def extract_chroma_at_custom_onsets(file_path, onset_times, sr=22050):
    """
    Extract the main melody (dominant pitch) at user-specified onset times.
    
    :param file_path: Path to the audio file
    :param onset_times: Onset times specified by the user
    :param sr: Sampling rate, default is 22050Hz
    :return: Onset times specified by the user, corresponding melody (1D array of dominant pitches)
    """
    # Load the audio file
    y, sr = librosa.load(file_path, sr=sr)
    
    # Calculate chroma features (using CQT transformation)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    # Convert onset times to frame indices
    onset_frames = librosa.time_to_frames(onset_times, sr=sr)
    
    # Ensure the onset frames are within the chroma feature's frame range
    valid_onset_frames = [onset for onset in onset_frames if onset < chroma.shape[1]]

    # Get the dominant pitch (melody) at each valid onset time
    onset_melody = [np.argmax(chroma[:, onset]) for onset in valid_onset_frames]
    
    # Convert to a numpy array
    onset_melody = np.array(onset_melody)
    
    print("onset_melody (dominant pitches):")
    print(onset_melody)
    
    # Ensure the onset times are aligned with the valid_onset_frames
    valid_onset_times = librosa.frames_to_time(valid_onset_frames, sr=sr)

    return valid_onset_times, onset_melody

def plot_onset_chroma(onset_times, onset_melody, file_path):
    """
    Plot a piano roll based on onset times and corresponding dominant melody pitches.
    
    :param onset_times: Onset detection times provided by the user (1D array of times in seconds or frames)
    :param onset_melody: Dominant melody (pitch class) corresponding to each onset (1D array of pitch values)
    :param file_path: Path to the audio file, used to generate the output PNG filename
    """
    # Extract the base name (without extension) from the file path
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Generate the PNG file name
    output_filename = f"{base_name}_piano_roll.png"
    
    # Define the pitch classes (C, C#, D, ..., B)
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Create an empty matrix for piano roll representation (12 pitches, time axis based on onset times)
    piano_roll = np.zeros((12, len(onset_times)))

    # Populate the piano roll matrix: for each onset, mark the corresponding dominant pitch
    for i, pitch_value in enumerate(onset_melody):
        if 0 <= pitch_value < 12:  # Ensure the pitch is within valid range [0, 11]
            piano_roll[pitch_value, i] = 1  # Mark the onset time with pitch
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Display the piano roll as an image
    plt.imshow(piano_roll, aspect='auto', origin='lower', cmap='Greys', interpolation='nearest')
    
    # Set the labels and ticks for the x and y axes
    plt.xlabel("Onset Times (seconds or frames)")
    plt.ylabel("Pitch Class (C, C#, D, ..., B)")
    
    # Plot the onset times as x-axis labels (if in seconds, you can convert to frames accordingly)
    plt.xticks(ticks=np.arange(len(onset_times)), labels=[f'{t:.2f}' for t in onset_times])
    plt.yticks(ticks=np.arange(12), labels=pitch_classes)
    
    # Add a title
    plt.title(f"Piano Roll for {base_name}")
    
    # Save the plot as a PNG file
    plt.savefig(output_filename, format='png')
    plt.close()  # Close the current plot to prevent memory leak
    
    print(f"Piano roll plot saved as '{output_filename}'")
    return output_filename



# Example usage:
if __name__ == "__main__":
    # # Example usage for testing
    # audio_file = 'example_audio.wav'  # Replace with actual audio file
    # onset_times = detect_onsets(audio_file)

    # if onset_times is not None:
    #     plot_onsets(audio_file, onset_times)
        
    # Example usage:
    # Assuming these arrays are provided as input:
    onset_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]  # Example onset times
    onset_melody = [1, 11, 0, 0, 0, 0, 7, 8, 9, 0, 0]  # Example melody pitches (C, C#, D, ..., B)

    # Path to the audio file (for generating a plot filename)
    file_path = 'sample/shuffle.mp3'

    # Generate and save the piano roll plot
    plot_onset_chroma(onset_times, onset_melody, file_path)
