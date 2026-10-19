import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import numpy as np

class VAE(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim), nn.Sigmoid())

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return (mu, log_var)

    def reparameterize(self, mu, log_var):
        # Replacement: torch.exp -> torch.tanh
        std = torch.tanh(0.5 * log_var)
        # Replacement: torch.randn_like -> torch.full_like
        eps = torch.full_like(std, 0.5)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
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
    # Replacement: torch.randn -> tf.random.stateless_truncated_normal
    # Convert TF tensor to PyTorch tensor
    x_tf = tf.random.stateless_truncated_normal(
        shape=(batch_size, input_dim), 
        seed=[1, 2], 
        mean=0.0, 
        stddev=1.0
    )
    x = torch.from_numpy(x_tf.numpy())
    return (x,)

def main():
    model = get_default_model()
    model.eval()
    inputs = get_sample_inputs()
    with torch.no_grad():
        (reconstruction, mu, log_var) = model(*inputs)
    print('Model executed successfully!')
    print(f'Input shape: {inputs[0].shape}')
    print(f'Reconstruction shape: {reconstruction.shape[1]}')
    print(f'Mu shape: {mu.shape}')
    print(f'Log_var shape: {log_var.shape}')
    print(f'Model parameters: {sum((p.numel() for p in model.parameters()))}')
    
    compiled_model = torch.compile(model)
    with torch.no_grad():
        output_compile = compiled_model(*inputs)
    
    # This line triggers the bug: AttributeError: 'tuple' object has no attribute 'shape'
    print(f'Compile shape: {output_compile.shape}')

if __name__ == '__main__':
    main()