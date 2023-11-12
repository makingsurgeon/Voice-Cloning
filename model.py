import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import fnmatch
import os
import random

#Used both ASVspoof and cv corpus data

DATASET_PATH = "/Users/zihuiouyang/Downloads/LA/ASVspoof2019_LA_train/flac"
LABEL_FILE_PATH = "/Users/zihuiouyang/Downloads/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
NUM_CLASSES = 2  # Number of classes (bonafide and spoof)
SAMPLE_RATE = 16000  # Sample rate of your audio files
DURATION = 5  # Duration of audio clips in seconds
N_MELS = 128  # Number of Mel frequency bins



def model():
    a = []
    for filename in os.listdir('/Users/zihuiouyang/Downloads/cv-corpus-15.0-delta-2023-09-08/en/clips'):
        if fnmatch.fnmatch(filename, '*.flac'):
            a.append(filename)
    a1 = random.sample(a,22800)
    labels = {}
    for i in range(len(a1)):
        file_name = a1[i]
        label = 1
        labels[file_name] = label
    X = []
    y = []

    max_time_steps = 250  # Define the maximum time steps for your model

    for file_name, label in labels.items():
        file_path = os.path.join("/Users/zihuiouyang/Downloads/cv-corpus-15.0-delta-2023-09-08/en/clips", file_name)

        # Load audio file using librosa
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

        # Extract Mel spectrogram using librosa
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Ensure all spectrograms have the same width (time steps)
        if mel_spectrogram.shape[1] < max_time_steps:
            mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, max_time_steps - mel_spectrogram.shape[1])), mode='constant')
        else:
            mel_spectrogram = mel_spectrogram[:, :max_time_steps]

        X.append(mel_spectrogram)
        y.append(label)
    
    labels1 = {}
    with open(LABEL_FILE_PATH, 'r') as label_file:
    lines = label_file.readlines()

    for line in lines:
        parts = line.strip().split()
        file_name = parts[1]
        if parts[-1] == "bonafide":
            continue
        label = 0
        labels1[file_name] = label 

    max_time_steps = 250  # Define the maximum time steps for your model

    for file_name, label in labels1.items():
        file_path = os.path.join(DATASET_PATH, file_name + ".flac")

        # Load audio file using librosa
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

        # Extract Mel spectrogram using librosa
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Ensure all spectrograms have the same width (time steps)
        if mel_spectrogram.shape[1] < max_time_steps:
            mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, max_time_steps - mel_spectrogram.shape[1])), mode='constant')
        else:
            mel_spectrogram = mel_spectrogram[:, :max_time_steps]

        X.append(mel_spectrogram)
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    y_encoded = to_categorical(y, NUM_CLASSES)
    split_index = int(0.8 * len(X))
    b = []
    for i in range(45600):
        b.append(i)
    b1 = random.sample(b,36480)
    mask=np.full(len(b),False,dtype=bool)
    mask[b1]=True
    X_train, X_val = X[mask], X[~mask]
    y_train, y_val = y_encoded[mask], y_encoded[~mask]

    input_shape = (N_MELS, X_train.shape[2])
    model = tf.keras.Sequential()
    model.add(LSTM(128,input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(48, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))