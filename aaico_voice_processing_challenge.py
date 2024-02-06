import librosa
import numpy as np
import time
import threading
import queue
import pickle

############################ sklearn model ############################

from sklearn.ensemble import RandomForestClassifier

# Define a simple classifier (you should train it on labeled data in a real scenario)
classifier = RandomForestClassifier()

############################

############################ transformers  model ############################

import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Tokenizer

# Load pre-trained transformer model and tokenizer
model_name = "facebook/wav2vec2-base-960h"
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

############################

########### PARAMETERS ###########
# DO NOT MODIFY
# Desired sample rate 16000 Hz
sample_rate = 16000
# Frame length
frame_length = 512

########### AUDIO FILE ###########
# DO NOT MODIFY
# Path to the audio file
audio_file = "test_aaico_challenge.wav"

# Read the audio file and resample it to the desired sample rate
audio_data, current_sample_rate = librosa.load(
    audio_file, 
    sr=sample_rate,
)
audio_data_int16 = (audio_data * 32767).astype(np.int16)
number_of_frames = len(audio_data_int16) // frame_length
audio_data_int16 = audio_data_int16[:number_of_frames * frame_length]
audio_duration = len(audio_data_int16) / sample_rate


########### STREAMING SIMULATION ###########
# DO NOT MODIFY
results = np.zeros(shape=(3, len(audio_data_int16)), dtype=np.int64)
# Detection mask lines are SENT TIME, LABEL, RECEIVE TIME. 
buffer = queue.Queue()
start_event = threading.Event()

def label_samples(list_samples_id, labels):
    receive_time = time.time_ns()
    results[1][list_samples_id] = labels
    results[2][list_samples_id] = receive_time

def notice_send_samples(list_samples_id):
    send_time = time.time_ns()
    results[0][list_samples_id] = send_time

def emit_data(): 
    time.sleep(.5)
    print('Start emitting')
    start_event.set()
    for i in range(0, number_of_frames):
        list_samples_id = np.arange(i*frame_length, (i+1)*frame_length)
        time.sleep(frame_length / sample_rate) # Simulate real time
        frame = audio_data_int16[list_samples_id]
        buffer.put(frame)
        notice_send_samples(list_samples_id)
    print('Stop emitting')

def extract_audio_features(frame):
    # Extract MFCC features (you might need to adjust parameters based on your data)
    mfcc_features = librosa.feature.mfcc(y=frame, sr=sample_rate, n_mfcc=13)
    return mfcc_features

def process_data():
    i = 0
    start_event.wait()
    print('Start processing')
    while i != number_of_frames:
        frame = buffer.get()

        ### TODO: YOUR CODE
        # Modify the feature extraction based on your needs
        features = extract_audio_features(frame)

        # Convert the features to text using the transformer's tokenizer
        input_text = tokenizer.batch_encode_plus(
            features.tolist(),
            padding=True,
            return_tensors="pt"
        )["input_values"]

        # Forward pass through the transformer model
        with torch.no_grad():
            logits = model(input_text).logits

        # Assuming binary classification, convert logits to probabilities
        probabilities = torch.softmax(logits, dim=1)
        labels = (probabilities[:, 0] > 0.5).long().numpy()
        ###

        list_samples_id = np.arange(i*frame_length, (i+1)*frame_length)
        label_samples(list_samples_id, labels)
        i += 1

    print('Stop processing')

    # Save the list to a file
    with open('results.pkl', 'wb') as file:
        pickle.dump(results, file)


if __name__ == "__main__":
    time_measurement = []

    thread_process = threading.Thread(target=process_data)
    thread_emit = threading.Thread(target=emit_data)

    thread_process.start()
    thread_emit.start()
