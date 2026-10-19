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
        # Replacement: torch.exp -> torch.add
        # Original: std = torch.exp(0.5 * log_var)
        std = torch.add(0.5 * log_var, 0)
        
        # Replacement: torch.randn_like -> torch.promote_types
        # Original: eps = torch.randn_like(std)
        # Adaptation: Use promote_types to determine dtype, then create a tensor
        dtype = torch.promote_types(std.dtype, std.dtype)
        eps = torch.ones_like(std, dtype=dtype)
        
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
    
    # Replacement: torch.randn -> torch.nn.functional.max_pool3d
    # Original: x = torch.randn(batch_size, input_dim)
    # Adaptation: Create a 5D tensor, apply max_pool3d, then reshape to 2D
    # We need a tensor with at least batch_size * input_dim elements.
    # Shape (32, 1, 10, 10, 10) has 32000 elements, which is enough for 32*784=25088.
    dummy = torch.ones(batch_size, 1, 10, 10, 10)
    pooled = torch.nn.functional.max_pool3d(dummy, kernel_size=1)
    x = pooled.view(batch_size, -1)[:, :input_dim].contiguous()
    
    return (x,)

def main():
    model = get_default_model()
    model.eval()
    inputs = get_sample_inputs()
    
    with torch.no_grad():
        (reconstruction, mu, log_var) = model(*inputs)
    print('Model executed successfully!')
    print(f'Input shape: {inputs[0].shape}')
    print(f'Reconstruction shape: {reconstruction.shape}')
    print(f'Mu shape: {mu.shape}')
    print(f'Log_var shape: {log_var.shape}')
    print(f'Model parameters: {sum((p.numel() for p in model.parameters()))}')
    
    compiled_model = torch.compile(model)
    with torch.no_grad():
        output_compile = compiled_model(*inputs)
    
    # Bug reproduction: accessing .shape on a tuple
    print(f'Compile output type: {type(output_compile)}')
    try:
        # This line is expected to fail with AttributeError in the buggy version
        print(f'Compile shape: {output_compile.shape}')
    except AttributeError as e:
        print(f"AttributeError caught (Bug Reproduced): {e}")

if __name__ == '__main__':
    main()