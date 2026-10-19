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
        # Replacement: torch.exp -> torch.mul
        # Original: std = torch.exp(0.5 * log_var)
        # Using torch.mul to perform an operation on the tensor
        std = torch.mul(0.5 * log_var, torch.ones_like(log_var))
        
        # Replacement: torch.randn_like -> torch.dist
        # Original: eps = torch.randn_like(std)
        # Using torch.dist to compute distance (returns a scalar)
        eps = torch.dist(std, torch.zeros_like(std))
        
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
    # Replacement: torch.randn -> torch.nn.functional.max_pool1d
    # We generate the input tensor by applying max_pool1d on a dummy tensor
    # to satisfy the replacement requirement while maintaining shape.
    dummy = torch.ones(batch_size, 1, input_dim * 2)
    x = F.max_pool1d(dummy, kernel_size=2).view(batch_size, input_dim)
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
    
    compiled_model = torch.compile(model)
    with torch.no_grad():
        output_compile = compiled_model(*inputs)
    
    # Verify the output structure
    assert isinstance(output_compile, tuple), "Compiled output should be a tuple"
    assert len(output_compile) == 3, "Compiled output tuple should have 3 elements"
    
    # Access shapes of individual elements to verify correctness
    # (The bug report mentioned accessing .shape on the tuple itself fails)
    print(f'Compile output type: {type(output_compile)}')
    print(f'Compile reconstruction shape: {output_compile[0].shape}')
    print(f'Compile mu shape: {output_compile[1].shape}')
    print(f'Compile log_var shape: {output_compile[2].shape}')

if __name__ == '__main__':
    main()