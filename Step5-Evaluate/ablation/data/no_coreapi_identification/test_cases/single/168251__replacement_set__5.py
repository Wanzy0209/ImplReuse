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
        # Replacement: torch.exp -> torch.sub
        # Original: std = torch.exp(0.5 * log_var)
        std = torch.sub(0.5 * log_var, 0)
        
        # Replacement: torch.randn_like -> torch.arange
        # Original: eps = torch.randn_like(std)
        # We generate a 1D range and reshape it to match std's shape to maintain tensor logic
        eps = torch.arange(0, std.numel(), device=std.device, dtype=std.dtype).reshape(std.shape)
        
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
    
    # Replacement: torch.randn -> torch.nn.functional.adaptive_max_pool3d
    # Original: x = torch.randn(batch_size, input_dim)
    # We create a dummy 5D tensor, apply adaptive_max_pool3d, and reshape to the expected input dimensions
    dummy_5d = torch.ones(batch_size, 1, 10, 10, 10)
    pooled = F.adaptive_max_pool3d(dummy_5d, (1, 1, input_dim))
    x = pooled.view(batch_size, input_dim)
    
    return (x,)

def main():
    model = get_default_model()
    model.eval()
    inputs = get_sample_inputs()
    
    # Eager execution
    with torch.no_grad():
        (reconstruction, mu, log_var) = model(*inputs)
        
    print('Model executed successfully!')
    print(f'Input shape: {inputs[0].shape}')
    print(f'Reconstruction shape: {reconstruction.shape}')
    print(f'Mu shape: {mu.shape}')
    print(f'Log_var shape: {log_var.shape}')
    print(f'Model parameters: {sum((p.numel() for p in model.parameters()))}')
    
    # Compiled execution
    compiled_model = torch.compile(model)
    with torch.no_grad():
        output_compile = compiled_model(*inputs)
        
    # Verify the output structure
    assert isinstance(output_compile, tuple), "Compiled output should be a tuple"
    assert len(output_compile) == 3, "Compiled output tuple should have 3 elements"
    
    (c_reconstruction, c_mu, c_log_var) = output_compile
    
    print(f'Compile Reconstruction shape: {c_reconstruction.shape}')
    print(f'Compile Mu shape: {c_mu.shape}')
    print(f'Compile Log_var shape: {c_log_var.shape}')
    
    # Verify shapes match
    assert reconstruction.shape == c_reconstruction.shape
    assert mu.shape == c_mu.shape
    assert log_var.shape == c_log_var.shape
    
    print("Test passed!")

if __name__ == '__main__':
    main()