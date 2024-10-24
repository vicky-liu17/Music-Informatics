import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from PIL import Image
import os
import time

# Function to process audio and generate normalized waveform plot
def generate_waveform_plot(file_path):
    try:
        # Load the audio file using pydub
        audio = AudioSegment.from_file(file_path)

        # If stereo, convert to mono
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples())
        sample_rate = audio.frame_rate

        # Normalize the samples to the range [-1, 1]
        max_val = np.max(np.abs(samples))  # Find the maximum absolute value in the samples
        if max_val > 0:  # Avoid division by zero
            samples = samples / max_val  # Normalize to [-1, 1]

        # Time axis for plotting
        time_axis = np.linspace(0, len(samples) / sample_rate, num=len(samples))

        # Get the base name of the file and remove its extension
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # Create a unique filename using the base name and a timestamp
        timestamp = int(time.time())  # Get current timestamp
        output_plot = f"{base_name}_{timestamp}_waveform.png"

        # Create the plot
        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, samples)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (Normalized)")
        plt.title(f"Waveform of {base_name} (Normalized)")

        # Save the plot as an image with the unique filename
        plt.savefig(output_plot)
        plt.close()

        # Return the path to the generated plot image
        return output_plot

    except Exception as e:
        print(f"Error processing the file: {e}")
        return None
