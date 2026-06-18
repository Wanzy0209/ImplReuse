import torch
import tensorflow as tf
import numpy as np

# Enable eager execution as the similar API to torch.compile
# This ensures operations execute immediately and return concrete values
tf.compat.v1.enable_eager_execution()

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
    model = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    return model

def get_sample_inputs():
    batch_size = 32
    input_dim = 784
    x = tf.random.normal((batch_size, input_dim))
    return (x,)

def main():
    model = get_default_model()
    inputs = get_sample_inputs()
    
    # Run in eager mode (enabled by the similar API)
    reconstruction, mu, log_var = model(inputs[0])
    print('Model executed successfully under eager execution!')
    print(f'Input shape: {inputs[0].shape}')
    print(f'Reconstruction shape: {reconstruction.shape}')
    print(f'Mu shape: {mu.shape}')
    print(f'Log_var shape: {log_var.shape}')
    
    # Reproduce the bug logic: accessing .shape on the tuple output
    # In the original PyTorch bug, torch.compile returns a tuple, and accessing .shape fails.
    # Here, with eager execution enabled, the model also returns a tuple.
    output_eager = model(inputs[0])
    
    try:
        # This line mimics the error in the original bug report: 'tuple' object has no attribute 'shape'
        print(f'Eager output shape: {output_eager.shape}')
        print("FAIL: Expected AttributeError but got a shape.")
    except AttributeError as e:
        print(f"SUCCESS: Caught expected AttributeError - {e}")
        print("This confirms that accessing .shape on a tuple output fails consistently.")

if __name__ == '__main__':
    main()