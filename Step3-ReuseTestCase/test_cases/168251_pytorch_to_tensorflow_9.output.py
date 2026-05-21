import torch
import tensorflow as tf
import numpy as np

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
        return (mu, log_var)

    def reparameterize(self, mu, log_var):
        std = tf.exp(0.5 * log_var)
        eps = tf.random.normal(shape=tf.shape(std))
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def call(self, x, training=False):
        # x = tf.reshape(x, [-1, self.input_dim]) # Reshape if necessary
        (mu, log_var) = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return (reconstruction, mu, log_var)

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
    
    # Eager execution
    (reconstruction, mu, log_var) = model(inputs[0], training=False)
    print('Model executed successfully!')
    print(f'Input shape: {inputs[0].shape}')
    print(f'Reconstruction shape: {reconstruction.shape}')
    print(f'Mu shape: {mu.shape}')
    print(f'Log_var shape: {log_var.shape}')
    print(f'Model parameters: {model.count_params()}')
    
    # Execution inside tf.name_scope (Similar API context)
    # Note: tf.name_scope is used for naming operations in the graph, 
    # analogous to how torch.compile wraps execution, though semantics differ.
    with tf.name_scope("vae_scope"):
        output_scope = model(inputs[0], training=False)
        
    # Verify behavior: accessing .shape on the tuple result
    # This reproduces the logic of the original bug report where the user
    # attempts to access .shape on a tuple output.
    try:
        print(f'Scope output shape: {output_scope.shape}')
    except AttributeError as e:
        print(f"AttributeError caught: {e}")
        print("Output is a tuple, unpacking is required to access shapes.")

if __name__ == '__main__':
    main()