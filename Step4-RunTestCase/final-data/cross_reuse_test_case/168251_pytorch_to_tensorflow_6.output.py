import torch
import numpy as np
import sys

# Handle environment dependency issues (e.g., libstdc++ version mismatch)
try:
    import tensorflow as tf
except ImportError as e:
    print("----------------------------------------------------------------")
    print("Error: Failed to import TensorFlow.")
    print(f"Details: {e}")
    print("----------------------------------------------------------------")
    print("This error is typically caused by a system library mismatch,")
    print("such as an outdated 'libstdc++' version (GLIBCXX_3.4.29 not found).")
    print("Please update your environment or system libraries to run this test.")
    print("----------------------------------------------------------------")
    sys.exit(1)

class VAE(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(hidden_dim, activation='relu')
        ])
        self.fc_mu = tf.keras.layers.Dense(latent_dim)
        self.fc_var = tf.keras.layers.Dense(latent_dim)
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(input_dim, activation='sigmoid')
        ])

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = tf.exp(0.5 * log_var)
        eps = tf.random.normal(shape=tf.shape(std))
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        x = tf.reshape(x, [-1, self.input_dim])
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var

def get_default_model():
    input_dim = 784
    hidden_dim = 400
    latent_dim = 20
    return VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)

def get_sample_inputs():
    batch_size = 32
    input_dim = 784
    x = tf.random.normal((batch_size, input_dim))
    return (x,)

def main():
    model = get_default_model()
    inputs = get_sample_inputs()

    # Eager execution
    reconstruction, mu, log_var = model(*inputs)
    print('Model executed successfully!')
    print(f'Input shape: {inputs[0].shape}')
    print(f'Reconstruction shape: {reconstruction.shape}')
    print(f'Mu shape: {mu.shape}')
    print(f'Log_var shape: {log_var.shape}')
    print(f'Model parameters: {model.count_params()}')

    # Using the similar API: tf.compat.v1.name_scope
    # This replaces the torch.compile context in the original test case.
    print("\n--- Testing with tf.compat.v1.name_scope ---")
    with tf.compat.v1.name_scope("vae_scope"):
        output_scope = model(*inputs)

    # Verify behavior: The original bug report implies accessing .shape on the output.
    # Since the model returns a tuple, accessing .shape on the tuple should fail.
    try:
        print(f'Scope output shape: {output_scope.shape}')
    except AttributeError as e:
        print(f"AttributeError: {e}")
        print("Output is a tuple, accessing .shape directly fails as expected.")

if __name__ == '__main__':
    main()