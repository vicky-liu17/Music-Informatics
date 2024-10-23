import tkinter as tk
from tkinter import filedialog, Label, Frame
from tkinter import font as tkfont
from PIL import Image, ImageTk  # For handling image display
import os
from audio_recognition import generate_waveform_plot  # Import the waveform plot function
from audio_onset_detection import detect_onsets, plot_onsets  # Import the onset detection logic

# Create the main window and set it to full screen
root = tk.Tk()
root.title("Audio Recognition GUI")
root.attributes('-fullscreen', True)  # Set the window to fullscreen

# Define custom fonts
title_font = tkfont.Font(family="Helvetica", size=18, weight="bold")
label_font = tkfont.Font(family="Arial", size=12)
button_font = tkfont.Font(family="Arial", size=14)

# Frame to organize the buttons and display
frame_top = Frame(root, bg="#F0F0F0")
frame_top.pack(pady=20)

# Title label
title_label = Label(root, text="Audio Recognition Tool", font=title_font, bg="#F0F0F0", fg="#333")
title_label.pack(pady=20)

# Display area for showing uploaded file names and result icon
display_frame = Frame(root, bg="#FFFFFF", relief="ridge", bd=2, width=800, height=100)
display_frame.pack(pady=10)

display_label = Label(display_frame, text="Upload files to display names here", font=label_font, bg="#FFFFFF", fg="#555")
display_label.place(relx=0.5, rely=0.5, anchor="center")

# Label to show the icon after recognition, and configure it to be responsive
icon_label = Label(root, bg="#F0F0F0")
icon_label.pack(pady=20, expand=True, fill='both')

# Styling and color effects for buttons
button_style = {"font": button_font, "bg": "#4CAF50", "fg": "black", "width": 20}
cancel_button_style = {"font": button_font, "bg": "#FF5733", "fg": "black", "width": 20}

# Global variables to store the paths to the audio files and plot images
sample_audio_path = None
practice_audio_path = None
sample_plot_path = None
practice_plot_path = None
sample_img = None
practice_img = None  # Keep separate references to each image
sample_onset_plot = None
practice_onset_plot = None  # Store paths for onset detection plots
waveform_buttons = []
onset_buttons = []
generated_files = []  # List to track all generated files

# Frame for buttons at the bottom
button_frame = Frame(root, bg="#F0F0F0", relief="ridge", bd=2)
button_frame.pack(side="bottom", fill="x")

# Function to handle file upload and generate the waveform
def upload_file(file_type):
    global sample_audio_path, practice_audio_path, sample_plot_path, practice_plot_path
    file_path = filedialog.askopenfilename(title=f"Select {file_type} Audio File", filetypes=[("Audio Files", "*.wav *.mp3")])
    if file_path:
        display_label.config(text=f"Processing {file_type}: {file_path.split('/')[-1]}")
        
        # Call the function from audio_recognition.py to generate waveform plot
        plot_path = generate_waveform_plot(file_path)
        
        if plot_path:
            display_label.config(text=f"{file_type} Processing Completed: {file_path.split('/')[-1]}")
            
            # Store the audio file path and plot path based on the file type
            if file_type == "Sample":
                sample_audio_path = file_path  # Store audio file path
                sample_plot_path = plot_path   # Store plot path (for visualization)
            elif file_type == "Practice":
                practice_audio_path = file_path  # Store audio file path
                practice_plot_path = plot_path   # Store plot path (for visualization)
            
            # Add the generated plot path to the list of generated files
            generated_files.append(plot_path)
            
            # Once both files are processed, show the buttons to display waveforms and onsets
            if sample_audio_path and practice_audio_path:
                display_label.config(text="Processing Completed for Both Files")
                show_waveform_buttons()
                show_onset_buttons()
        else:
            display_label.config(text=f"Error processing {file_type} file!")

# Function to display the "Sample" or "Practice" waveform
def display_waveform(file_type):
    global sample_plot_path, practice_plot_path, sample_img, practice_img
    if file_type == "Sample" and sample_plot_path:
        img = Image.open(sample_plot_path)
        img = img.resize((850, 450))  # Resize the image to a larger size for full screen
        sample_img = ImageTk.PhotoImage(img)  # Keep a reference to the sample image
        icon_label.config(image=sample_img)
    elif file_type == "Practice" and practice_plot_path:
        img = Image.open(practice_plot_path)
        img = img.resize((850, 450))  # Resize the image to a larger size for full screen
        practice_img = ImageTk.PhotoImage(img)  # Keep a reference to the practice image
        icon_label.config(image=practice_img)
    else:
        display_label.config(text=f"Error displaying {file_type} waveform.")

# Function to detect and display onsets for Sample or Practice
def display_onsets(file_type):
    global sample_onset_plot, practice_onset_plot
    file_path = sample_audio_path if file_type == "Sample" else practice_audio_path  # Use audio file path here
    
    # Perform onset detection
    onset_times = detect_onsets(file_path)  # Pass audio file path
    
    # Generate the onset plot
    plot_path = plot_onsets(file_path, onset_times)  # Use audio file path for plotting onsets
    generated_files.append(plot_path)
    if file_type == "Sample":
        sample_onset_plot = plot_path
    else:
        practice_onset_plot = plot_path
    
    # Display the generated onset plot
    if plot_path:
        img = Image.open(plot_path)
        img = img.resize((850, 450))  # Resize the image
        img_tk = ImageTk.PhotoImage(img)
        icon_label.config(image=img_tk)
        icon_label.image = img_tk  # Keep a reference to prevent garbage collection

# Function to create buttons for showing waveforms once both are processed
def show_waveform_buttons():
    global waveform_buttons

    # Clear any existing buttons before placing new ones
    for button in waveform_buttons:
        button.grid_forget()

    # Create buttons to display either the "Sample" or "Practice" waveform
    button_sample_waveform = tk.Button(button_frame, text="Show Sample Waveform", command=lambda: display_waveform("Sample"), **button_style)
    button_sample_waveform.grid(row=0, column=0, padx=10, pady=10)  # Use grid to fix placement
    waveform_buttons.append(button_sample_waveform)

    button_practice_waveform = tk.Button(button_frame, text="Show Practice Waveform", command=lambda: display_waveform("Practice"), **button_style)
    button_practice_waveform.grid(row=0, column=1, padx=10, pady=10)  # Use grid to fix placement
    waveform_buttons.append(button_practice_waveform)

# Function to create buttons for showing onsets once both are processed
def show_onset_buttons():
    global onset_buttons

    # Clear any existing buttons before placing new ones
    for button in onset_buttons:
        button.grid_forget()

    # Create button to display the "Sample" onsets
    button_sample_onset = tk.Button(button_frame, text="Show Sample Onsets", command=lambda: display_onsets("Sample"), **button_style)
    button_sample_onset.grid(row=1, column=0, padx=10, pady=10)  # Use grid to fix placement
    onset_buttons.append(button_sample_onset)

    # Create button to display the "Practice" onsets
    button_practice_onset = tk.Button(button_frame, text="Show Practice Onsets", command=lambda: display_onsets("Practice"), **button_style)
    button_practice_onset.grid(row=1, column=1, padx=10, pady=10)  # Use grid to fix placement
    onset_buttons.append(button_practice_onset)

# Function to reset the interface
def cancel_upload():
    global sample_audio_path, practice_audio_path, sample_plot_path, practice_plot_path, waveform_buttons, onset_buttons, sample_img, practice_img

    # Reset file paths and remove any displayed images
    sample_audio_path = None
    practice_audio_path = None
    sample_plot_path = None
    practice_plot_path = None
    sample_img = None
    practice_img = None  # Reset image references
    icon_label.config(image='')  # Clear the displayed image
    
    # Reset the display label
    display_label.config(text="Upload files to display names here")

    # Hide and remove the waveform buttons if they exist
    for button in waveform_buttons:
        button.grid_forget()
    waveform_buttons.clear()

    # Hide and remove the onset buttons if they exist
    for button in onset_buttons:
        button.grid_forget()
    onset_buttons.clear()
    
# Function to delete all generated PNG files
def delete_generated_files():
    global generated_files
    for file in generated_files:
        if os.path.exists(file):
            os.remove(file)
    generated_files.clear()

# Create two buttons: one for Sample, one for Practice
button_sample = tk.Button(frame_top, text="Upload Sample", command=lambda: upload_file("Sample"), **button_style)
button_sample.grid(row=0, column=0, padx=20)

button_practice = tk.Button(frame_top, text="Upload Practice", command=lambda: upload_file("Practice"), **button_style)
button_practice.grid(row=0, column=1, padx=20)

# Create a "Cancel" button to reset the state
button_cancel = tk.Button(frame_top, text="Cancel", command=cancel_upload, **cancel_button_style)
button_cancel.grid(row=0, column=2, padx=20)

# Function to exit full screen and delete generated files
def on_close():
    delete_generated_files()  # Delete all PNG files before closing
    root.destroy()  # Close the application

# Bind the window close event to delete the files
root.protocol("WM_DELETE_WINDOW", on_close)

# Bind the 'Esc' key to exit fullscreen mode
root.bind("<Escape>", lambda event: root.attributes('-fullscreen', False))

# Run the application
root.mainloop()