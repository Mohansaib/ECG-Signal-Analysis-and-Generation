import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Model, losses

# Load the training data (multiple people's amplitude data)
train_df = pd.read_csv(r'C:\\dinesh\\dinesh\\ed_final\\ecg.csv', header=None)
train_data = train_df.values

# Normalize the training data   
min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)
train_data = (train_data - min_val) / (max_val - min_val)
train_data = tf.cast(train_data, dtype=tf.float32)

# Define the autoencoder model class
class Detector(Model):
    def __init__(self):
        super(Detector, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(train_data.shape[1],)),
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(train_data.shape[1], activation='sigmoid')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Instantiate the model
autoencoder = Detector()
autoencoder.compile(optimizer='adam', loss='mae')

# Train the autoencoder model
autoencoder.fit(train_data, train_data, epochs=20, batch_size=512, validation_split=0.2)

# Load the new data (single person's amplitude data)
new_data_df = pd.read_csv(r'C:\\dinesh\\dinesh\\ed_final\\output_data.csv', header=None)
new_data = new_data_df.values

# Ensure the correct data format and normalization
new_data = new_data.astype(np.float32)  # Convert entire array to float32
new_data = (new_data - min_val) / (max_val - min_val)  # Normalize using min_val and max_val from training data

# Check and adjust shape to match model input
if new_data.shape[1] != train_data.shape[1]:  # Ensure the number of features matches
    raise ValueError(f"Expected {train_data.shape[1]} features, but got {new_data.shape[1]} features in new_data.")

# Reshape to match model's input shape (assuming batch size of 1)
new_data = np.reshape(new_data, (new_data.shape[0], train_data.shape[1]))  # Reshape to (num_samples, num_features)

# Predict using the trained model
reconstructed = autoencoder(new_data)
loss = losses.mae(reconstructed, new_data)

# Calculate threshold for anomaly detection (using train data)
train_reconstructed = autoencoder(train_data)
train_loss = losses.mae(train_reconstructed, train_data)
threshold = np.mean(train_loss) + np.std(train_loss)

# Determine if the new data indicates a potential cardiac arrest
anomalies = loss > threshold

# Output the results
if anomalies.numpy().any():
    print("The given data indicates some abnormality in ECG kindly concent concerned specialist.")
else:
    print("The given data is normal.")
