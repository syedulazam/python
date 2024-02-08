import librosa
import numpy as np
import time
import threading
import queue
import pickle

# Desired sample rate 16000 Hz
sample_rate = 16000
# Frame length
frame_length = 512

# Path to the audio file
audio_file = "test_aaico_challenge.wav"

# Read the audio file and resample it to the desired sample rate
audio_data, _ = librosa.load(audio_file, sr=sample_rate)

# Split the provided text into sentences
sentences = [
    "engine 3, this is bravo team. We have got heavy smoking flames visible from Charlie's side. We are going for a defensive attack. Over.",
    "Galactic temperature. This is bravo team. Visibility 0. Lots of heat. We are going to hit charlie hard from the exterior then make entry from a primary search. Over.",
    "Galactic battery. Keep an eye on that roof. Looks sketchy. Engine 3, better charge the host. Over.",
    "Acknowledged. We have got a 2.5 inch line ready for a blitz attack. We will knock down the bulk of the fire before we go in. Galactic oxygen. Pass is active and we are staying on the coms. Going in for the attack now. Stay sharp out there. Over.",
    "Bravo going in. Over."
]

# Define commands and non-commands based on provided sentences
commands = [1, 1, 1, 1, 1]  # All sentences are considered as commands
non_commands = [0, 0, 0, 0, 0]  # No non-command sentences in this example

# Combine commands and non-commands into a single list
labels = commands + non_commands

# Create a list to store the labeled samples
labeled_samples = []

# Simulate the streaming and labeling process
for i, sentence in enumerate(sentences):
    # Simulate real-time processing
    time.sleep(frame_length / sample_rate)

    # Simulate extracting features from the audio frame
    frame_start = i * frame_length
    frame_end = (i + 1) * frame_length
    frame_audio_data = audio_data[frame_start:frame_end]

    # Append the labeled sample to the list
    labeled_samples.append((frame_audio_data, labels[i]))

# Save the labeled samples to a file
with open('label_samples.pkl', 'wb') as file:
    pickle.dump(labeled_samples, file)
