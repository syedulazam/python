import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle

def load_labeled_data():
    # Load your labeled dataset (replace this with your actual loading mechanism)
    with open('label_samples.pkl', 'rb') as file:
        labeled_samples = pickle.load(file)

    features = np.array([sample[0] for sample in labeled_samples])
    labels = np.array([sample[1] for sample in labeled_samples])

    return features, labels

def build_model(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    features, labels = load_labeled_data()

    # Split data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Feature shape for the neural network
    input_shape = features.shape[1:]

    model = build_model(input_shape)

    # Define callbacks (early stopping and model checkpoint)
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', save_best_only=True)
    ]

    # Train the model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=20,  # Adjust as needed
        batch_size=32,  # Adjust as needed
        callbacks=callbacks
    )

    # Save the trained model
    model.save('final_model.h5')

    # Save training history for analysis or plotting
    with open('training_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)

if __name__ == "__main__":
    train_model()
