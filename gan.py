import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load your dataset
# Replace 'your_dataset.csv' with the path to your CSV file
data = pd.read_csv('ecg.csv')

# Assuming the last column is the label and the rest are features
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values   # Labels

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Implementing the GAN

import tensorflow as tf
from tensorflow.keras import layers

def build_generator(input_dim):
    model = tf.keras.Sequential([
        layers.Dense(128, input_dim=input_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(256),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(X_train.shape[1], activation='tanh')  # Match feature dimension
    ])
    return model

def build_discriminator(input_dim):
    model = tf.keras.Sequential([
        layers.Dense(256, input_dim=input_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(128),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

input_dim = 100  # Dimension of random noise input to generator

# Create the GAN components
generator = build_generator(input_dim)
discriminator = build_discriminator(X_train.shape[1])

# Compile the discriminator
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create and compile the GAN
discriminator.trainable = False
gan_input = layers.Input(shape=(input_dim,))
gan_output = discriminator(generator(gan_input))
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

#Training the GAN

def train_gan(generator, discriminator, gan, epochs, batch_size):
    for epoch in range(epochs):
        # Train discriminator with real samples
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_samples = X_train[idx]
        real_labels = np.ones((batch_size, 1))

        # Train discriminator with fake samples
        noise = np.random.normal(0, 1, (batch_size, input_dim))
        fake_samples = generator.predict(noise)
        fake_labels = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator
        noise = np.random.normal(0, 1, (batch_size, input_dim))
        valid_labels = np.ones((batch_size, 1))  # Generator tries to trick the discriminator

        g_loss = gan.train_on_batch(noise, valid_labels)

        # Print progress
        print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100*d_loss[1]}] [G loss: {g_loss}]")
        plt.plot(np.arange(140), real_samples[0])
        plt.grid()
        plt.title(' ECG')
        plt.show()

# Train the GAN
train_gan(generator, discriminator, gan, epochs=10, batch_size=64)

#Train the Classifier and Calculate Accuracy

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Generate synthetic data
noise = np.random.normal(0, 1, (1000, input_dim))
synthetic_data = generator.predict(noise)

# Combine real and synthetic data
X_augmented = np.vstack((X_train, synthetic_data))
y_augmented = np.hstack((y_train, np.ones(synthetic_data.shape[0])))  # Assuming synthetic data is labeled as positive

# Train the classifier
classifier = RandomForestClassifier()
classifier.fit(X_augmented, y_augmented)

# Evaluate the classifier
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Classifier Accuracy: {accuracy * 100:.2f}%")