import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os

# DTW comparison and alignment plot generation
def compare_onsets_dtw(sample_onset_times, practice_onset_times, sample_onset_chroma, practice_onset_chroma, output_dir="output"):
    # Convert to numpy arrays and flatten to ensure they are 1-D
    sample_onset_times = np.ravel(np.array(sample_onset_times))
    practice_onset_times = np.ravel(np.array(practice_onset_times))
    sample_onset_chroma = np.ravel(np.array(sample_onset_chroma))
    practice_onset_chroma = np.ravel(np.array(practice_onset_chroma))
    
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

def compare_onsets_dtw_2d(sample_onset_times, sample_onset_strength, practice_onset_times, practice_onset_strength, output_dir="output"):
    """
    Compare sample and practice onset times and strength using 2D DTW, analyze similarity, and generate alignment plot.

    Parameters:
    - sample_onset_times (list or np.ndarray): Detected onset times for the sample.
    - sample_onset_strength (list or np.ndarray): Detected onset strength for the sample.
    - practice_onset_times (list or np.ndarray): Detected onset times for the practice.
    - practice_onset_strength (list or np.ndarray): Detected onset strength for the practice.
    - output_dir (str): Directory where the output plot will be saved.

    Returns:
    - dtw_distance (float): The calculated DTW distance between the two onset sequences.
    - dtw_plot_path (str): Path to the saved DTW alignment plot image.
    """
    
    # Convert to numpy arrays and flatten to ensure they are 1-D
    sample_onset_times = np.ravel(np.array(sample_onset_times))
    sample_onset_strength = np.ravel(np.array(sample_onset_strength))
    practice_onset_times = np.ravel(np.array(practice_onset_times))
    practice_onset_strength = np.ravel(np.array(practice_onset_strength))

    # Combine onset times and onset strength into 2D feature vectors
    sample_features = np.column_stack((sample_onset_times, sample_onset_strength))
    practice_features = np.column_stack((practice_onset_times, practice_onset_strength))
    
    # Check and print the shapes to confirm they are 1-D
    print("Sample features shape inside function:", sample_features.shape)
    print("Practice features shape inside function:", practice_features.shape)

    # Compute DTW similarity on the 2D features
    dtw_distance, path = fastdtw(sample_features, practice_features, dist=euclidean)
    print(f"DTW Distance (2D): {dtw_distance}")

    # Plot alignment path and generate an image
    plt.figure(figsize=(10, 6))

    # Plot onset times and strength for sample and practice
    plt.plot(sample_onset_times, sample_onset_strength, label="Sample Onsets", color='blue', marker='o', linestyle='-')
    plt.plot(practice_onset_times, practice_onset_strength, label="Practice Onsets", color='red', marker='x', linestyle='-')

    # Show DTW alignment path
    for (i, j) in path:
        plt.plot([sample_onset_times[i], practice_onset_times[j]],
                 [sample_onset_strength[i], practice_onset_strength[j]], 
                 color='gray', linestyle='--', alpha=0.6)

    # Add labels and save the plot
    plt.xlabel("Onset Time (s)")
    plt.ylabel("Onset Strength (normalized)")
    plt.title(f"2D DTW Alignment Between Sample and Practice Onsets\nDTW Distance: {dtw_distance:.2f}")
    plt.legend()
    plt.grid(True)

    # Output plot path
    dtw_plot_path = os.path.join(output_dir, "dtw_onsets_alignment_2d.png")
    plt.savefig(dtw_plot_path)
    plt.close()

    print(f"DTW alignment plot saved as '{dtw_plot_path}'")
    return dtw_distance, dtw_plot_path