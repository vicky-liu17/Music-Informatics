import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os

def create_chroma_sequence(onset_times, chroma, step_size=0.2):
    """
    Creates a 1D array based on onset_times and chroma values, where
    time is divided into steps of `step_size` seconds. The onset times
    will have the corresponding chroma values, and other times will have 0.

    Args:
        onset_times (np.ndarray): 1D array of onset times in seconds.
        chroma (np.ndarray): 1D array of chroma values corresponding to onset times.
        step_size (float): The size of each time step in seconds. Default is 0.2.

    Returns:
        np.ndarray: A 1D array where chroma values are placed at onset times.
    """
    # Determine the maximum time and add a small buffer to avoid index out of bounds
    max_time = max(onset_times)
    num_steps = int(np.ceil(max_time / step_size)) + 1  # Add 1 to handle edge cases

    # Create an array filled with zeros
    chroma_sequence = np.zeros(num_steps)

    # Fill the array with chroma values at the correct onset indices
    for onset, chroma_value in zip(onset_times, chroma):
        index = int(onset / step_size)  # Find the index for each onset time
        if index < num_steps:  # Check to avoid index out of bounds
            chroma_sequence[index] = chroma_value  # Set the corresponding chroma value

    return chroma_sequence
    

# DTW comparison and alignment plot generation
def compare_onsets_dtw(sample_onset_times, practice_onset_times, sample_onset_chroma, practice_onset_chroma, output_dir="output"):
    # Convert to numpy arrays and flatten to ensure they are 1-D
    sample_onset_times = np.ravel(np.array(sample_onset_times))
    practice_onset_times = np.ravel(np.array(practice_onset_times))
    sample_onset_chroma = np.ravel(np.array(sample_onset_chroma))
    practice_onset_chroma = np.ravel(np.array(practice_onset_chroma))
    
    sample_sequence = create_chroma_sequence(sample_onset_times,sample_onset_chroma)
    practice_sequence = create_chroma_sequence(practice_onset_times,practice_onset_chroma)
    print(sample_sequence)
    print(practice_sequence)
    
    # Compute DTW similarity
    dtw_distance, path = fastdtw(sample_sequence, practice_sequence, dist=lambda x, y: euclidean([x], [y]))
    print(f"DTW Distance: {dtw_distance}")

    # Plot alignment path and generate an image
    plt.figure(figsize=(10, 6))

    # Plot onset times for sample and practice
    plt.plot(sample_onset_times, label="Sample Onsets", color='blue', marker='o')
    plt.plot(practice_onset_times, label="Practice Onsets", color='red', marker='x')

    # Show DTW alignment path
    for (i, j) in path:
        if i < len(sample_onset_times) and j < len(practice_onset_times):  # Ensure indices are within bounds
            plt.plot([i, j], [sample_onset_times[i], practice_onset_times[j]], color='gray', linestyle='--')


    # Add labels and save the plot
    plt.xlabel("Index")
    plt.ylabel("Notes")
    plt.title(f"DTW Alignment Between Sample and Practice Onsets\nDTW Distance: {dtw_distance:.2f}")
    plt.legend()
    plt.grid(True)

    # Output plot path
    dtw_plot_path = os.path.join(output_dir, "dtw_onsets_alignment.png")
    plt.savefig(dtw_plot_path)
    plt.close()

    print(f"DTW alignment plot saved as '{dtw_plot_path}'")
    return dtw_distance, dtw_plot_path

def compare_onsets_dtw_weighted(sample_onset_times, sample_onset_strength, sample_onset_chroma, 
                                practice_onset_times, practice_onset_strength, practice_onset_chroma, 
                                alpha=0.1, output_dir="output"):
    """
    Compare sample and practice onset times and chroma using weighted 1D DTW with transformed sequences.
    """
    # Convert to numpy arrays and flatten to ensure they are 1-D
    sample_onset_times = np.ravel(np.array(sample_onset_times))
    sample_onset_strength = np.ravel(np.array(sample_onset_strength))
    practice_onset_times = np.ravel(np.array(practice_onset_times))
    practice_onset_strength = np.ravel(np.array(practice_onset_strength))
    sample_onset_chroma = np.ravel(np.array(sample_onset_chroma))
    practice_onset_chroma = np.ravel(np.array(practice_onset_chroma))
    
    # Transform sequences and flatten them to ensure 1D
    sample_sequence = np.ravel(create_chroma_sequence(sample_onset_times, sample_onset_chroma))
    practice_sequence = np.ravel(create_chroma_sequence(practice_onset_times, practice_onset_chroma))
    sample_strength_sequence = np.ravel(create_chroma_sequence(sample_onset_times, sample_onset_strength))
    practice_strength_sequence = np.ravel(create_chroma_sequence(practice_onset_times, practice_onset_strength))
    
   

    # Compute DTW without weighted distance to get the alignment path
    _, path = fastdtw(sample_sequence, practice_sequence, dist=euclidean)

    # Calculate weighted DTW distance using the path
    dtw_distance = 0
    for (i, j) in path:
        avg_strength = (sample_strength_sequence[i] + practice_strength_sequence[j]) / 2  # Average strength
        distance = euclidean(sample_sequence[i], practice_sequence[j])  # Use directly without wrapping in a list
        weighted_distance = distance * (1 + alpha * avg_strength)
        dtw_distance += weighted_distance

    print(f"DTW Distance (Weighted 1D): {dtw_distance}")

    # Plot alignment path and generate an image
    plt.figure(figsize=(10, 6))

    # Plot onset times and chroma for sample and practice
    plt.plot(sample_onset_times, sample_sequence, label="Sample Sequence", color='blue', marker='o', linestyle='-')
    plt.plot(practice_onset_times, practice_sequence, label="Practice Sequence", color='red', marker='x', linestyle='-')

    # Show DTW alignment path
    for (i, j) in path:
        if i < len(sample_onset_times) and j < len(practice_onset_times):  # Ensure indices are within bounds
            plt.plot([sample_onset_times[i], practice_onset_times[j]], 
                     [sample_sequence[i], practice_sequence[j]], 
                     color='gray', linestyle='--', alpha=0.6)

    # Add labels and save the plot
    plt.xlabel("Onset Time (s)")
    plt.ylabel("Chroma/Strength Value")
    plt.title(f"Weighted 1D DTW Alignment Between Sample and Practice Onsets\nDTW Distance: {dtw_distance:.2f}")
    plt.legend()
    plt.grid(True)

    # Output plot path
    dtw_plot_path = os.path.join(output_dir, "dtw_onsets_alignment_weighted_1d_transformed.png")
    plt.savefig(dtw_plot_path)
    plt.close()

    print(f"DTW alignment plot saved as '{dtw_plot_path}'")
    return dtw_distance, dtw_plot_path
