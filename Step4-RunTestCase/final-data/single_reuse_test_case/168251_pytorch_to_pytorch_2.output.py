import torch
import torch.nn as nn
import torch.nn.functional as F

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
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
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
    x = torch.randn(batch_size, input_dim)
    return (x,)

def main():
    model = get_default_model()
    model.eval()
    inputs = get_sample_inputs()
    
    with torch.no_grad():
        reconstruction, mu, log_var = model(*inputs)
    
    print('Model executed successfully!')
    print(f'Input shape: {inputs[0].shape}')
    print(f'Reconstruction shape: {reconstruction.shape}')
    print(f'Mu shape: {mu.shape}')
    print(f'Log_var shape: {log_var.shape}')
    
    # Adapted test case for torch.prod
    # The original bug involved accessing .shape on a tuple returned by torch.compile.
    # Here we verify that torch.prod returns a Tensor with a valid .shape attribute.
    
    # Test 1: torch.prod on the reconstruction tensor
    prod_recon = torch.prod(reconstruction)
    print(f'Prod Reconstruction shape: {prod_recon.shape}')
    assert prod_recon.shape == torch.Size([]), "Expected scalar output for full reduction"
    
    # Test 2: torch.prod on the mu tensor
    prod_mu = torch.prod(mu)
    print(f'Prod Mu shape: {prod_mu.shape}')
    assert prod_mu.shape == torch.Size([]), "Expected scalar output for full reduction"
    
    # Test 3: torch.prod on the log_var tensor
    prod_log_var = torch.prod(log_var)
    print(f'Prod Log_var shape: {prod_log_var.shape}')
    assert prod_log_var.shape == torch.Size([]), "Expected scalar output for full reduction"
    
    # Test 4: torch.prod with specific dimension
    prod_recon_dim = torch.prod(reconstruction, dim=0)
    print(f'Prod Reconstruction (dim=0) shape: {prod_recon_dim.shape}')
    assert prod_recon_dim.shape == torch.Size([784]), "Expected shape [784] for reduction on dim 0"

    print('torch.prod test passed!')

if __name__ == '__main__':
    main()