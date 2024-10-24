import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

def piano_note_recognition(file_path):
    # Load the audio file
    data, Fs = librosa.load(file_path, sr=None)
    
    # Get the note windows
    w = notewindows(data, Fs)
    
    # Initialize lists for storing pitch and onset times for the piano roll
    onsets = []
    pitches = []
    
    # How many notes in this sample
    num_notes = len(w) - 1
    
    for i in range(num_notes):
        # Take the window for the current note
        cur_note = data[w[i]:w[i+1]]
        len_note = len(cur_note)
        
        # FFT calculation and magnitude spectrum
        cur_fft = np.abs(np.fft.fft(cur_note))
        cur_fft = cur_fft[:len_note // 2 + 1]
        
        # Find the frequency with the highest magnitude
        I = np.argmax(cur_fft)
        I = I * Fs / len_note
        
        print(f'Frequency of note {i+1}: {I} Hz')
        
        # Bandpass filtering around the dominant frequency
        data_filtered = bandpass_filter(data, I - 0.1, I + 0.1, Fs)
        
        # Detect pitch and octave
        pitch, octave = findpitch(I)
        
        # Store the pitch and onset time for the piano roll
        onsets.append(w[i] / Fs)  # Convert to seconds
        pitches.append(pitch + 12 * (octave + 4))  # Convert to MIDI note number

    # Generate the piano roll plot
    plot_piano_roll(onsets, pitches, num_notes)

def notewindows(data, Fs):
    # Wrapper for noteparse to add the beginning and end points
    divs = noteparse(data)
    
    # Massage intervals to include the start and end of the signal
    d2 = [0] + divs + [len(data)]
    
    w = [(d2[i] + d2[i+1]) // 2 for i in range(0, len(d2)-1, 2)]
    return w

def noteparse(data):
    len_data = len(data)
    
    # Set thresholds to determine when notes start and stop
    threshup = 0.2 * np.max(data)
    threshdown = 0.04 * np.max(data)
    
    divs = []
    quiet = True  # flag to track noisy/quiet segments
    
    for i in range(50, len_data - 50):
        if quiet:
            if np.max(np.abs(data[i-50:i+50])) > threshup:
                quiet = False  # Start of a note
                divs.append(i)
        else:
            if np.max(np.abs(data[i-50:i+50])) < threshdown:
                quiet = True  # End of a note
                divs.append(i)
    
    return divs

def findpitch(freq):
    # Determine octave and pitch given a frequency
    octave = helpfindoctave(freq, 0)
    pitch = choosepitch(freq / 2**octave)
    return pitch, octave

def helpfindoctave(f, o):
    if 254.284 <= f <= 508.5675:
        return o
    elif f < 254.284:
        return helpfindoctave(2 * f, o - 1)
    else:
        return helpfindoctave(f / 2, o + 1)

def choosepitch(f):
    if 254.284 <= f < 269.4045:
        return 1  # C
    elif 269.4045 <= f < 285.424:
        return 2  # C#
    elif 285.424 <= f < 302.396:
        return 3  # D
    elif 302.396 <= f < 320.3775:
        return 4  # D#
    elif 320.3775 <= f < 339.428:
        return 5  # E
    elif 339.428 <= f < 359.611:
        return 6  # F
    elif 359.611 <= f < 380.9945:
        return 7  # F#
    elif 380.9945 <= f < 403.65:
        return 8  # G
    elif 403.65 <= f < 427.6525:
        return 9  # G#
    elif 427.6525 <= f < 453.082:
        return 10  # A
    elif 453.082 <= f < 480.0235:
        return 11  # A#
    elif 480.0235 <= f < 508.567:
        return 12  # B
    else:
        raise ValueError('Frequency outside of acceptable range')

def bandpass_filter(data, low_freq, high_freq, Fs):
    # Design bandpass filter
    nyquist = 0.5 * Fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(N=4, Wn=[low, high], btype='band')
    return lfilter(b, a, data)

def plot_piano_roll(onsets, pitches, num_notes):
    # Create a piano roll plot
    plt.figure(figsize=(10, 6))
    
    for i in range(num_notes):
        plt.barh(pitches[i], 0.1, left=onsets[i], height=0.9)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('MIDI Note Number')
    plt.title('Piano Roll')
    plt.grid(True)
    plt.show()

# Example usage:
piano_note_recognition('sample/bee.mp3')
