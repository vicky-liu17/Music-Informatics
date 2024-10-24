import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os

# DTW comparison and alignment plot generation
def compare_onsets_dtw(sample_onset_times, practice_onset_times, output_dir="output"):
    """
    Compare sample and practice onset times using DTW, analyze similarity, and generate alignment plot.

    Parameters:
    - sample_onset_times (list or np.ndarray): Detected onset times for the sample.
    - practice_onset_times (list or np.ndarray): Detected onset times for the practice.
    - output_dir (str): Directory where the output plot will be saved.

    Returns:
    - dtw_distance (float): The calculated DTW distance between the two onset time sequences.
    - dtw_plot_path (str): Path to the saved DTW alignment plot image.
    """
    
    # Convert to numpy arrays and flatten to ensure they are 1-D
    sample_onset_times = np.ravel(np.array(sample_onset_times))
    practice_onset_times = np.ravel(np.array(practice_onset_times))
    
    # Check and print the shapes to confirm they are 1-D
    print("Sample onset shape inside function:", sample_onset_times.shape)
    print("Practice onset shape inside function:", practice_onset_times.shape)

    # Compute DTW similarity
    dtw_distance, path = fastdtw(sample_onset_times, practice_onset_times, dist=lambda x, y: euclidean([x], [y]))
    print(f"DTW Distance: {dtw_distance}")

    # Plot alignment path and generate an image
    plt.figure(figsize=(10, 6))

    # Plot onset times for sample and practice
    plt.plot(sample_onset_times, label="Sample Onsets", color='blue', marker='o')
    plt.plot(practice_onset_times, label="Practice Onsets", color='red', marker='x')

    # Show DTW alignment path
    for (i, j) in path:
        plt.plot([i, j], [sample_onset_times[i], practice_onset_times[j]], color='gray', linestyle='--')

    # Add labels and save the plot
    plt.xlabel("Index")
    plt.ylabel("Onset Time (s)")
    plt.title(f"DTW Alignment Between Sample and Practice Onsets\nDTW Distance: {dtw_distance:.2f}")
    plt.legend()
    plt.grid(True)

    # Output plot path
    dtw_plot_path = os.path.join(output_dir, "dtw_onsets_alignment.png")
    plt.savefig(dtw_plot_path)
    plt.close()

    print(f"DTW alignment plot saved as '{dtw_plot_path}'")
    return dtw_distance, dtw_plot_path
