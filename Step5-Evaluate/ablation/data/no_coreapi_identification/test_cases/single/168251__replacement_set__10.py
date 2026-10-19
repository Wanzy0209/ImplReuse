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
        # Replacement: torch.exp -> torch.nn.functional.softplus
        # Original: std = torch.exp(0.5 * log_var)
        std = F.softplus(0.5 * log_var)
        
        # Replacement: torch.randn_like -> torch.median
        # Original: eps = torch.randn_like(std)
        # torch.median returns (values, indices). We take values.
        eps, _ = torch.median(std)
        
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
    
    # Replacement: torch.randn -> torch.nn.functional.cosine_embedding_loss
    # Original: x = torch.randn(batch_size, input_dim)
    # To keep the test runnable, we generate inputs for the loss function
    # and then expand the result to match the expected input shape for the VAE.
    input1 = torch.randn(batch_size, input_dim)
    input2 = torch.randn(batch_size, input_dim)
    target = torch.ones(batch_size)
    
    # cosine_embedding_loss returns a scalar by default, or (batch_size,) with reduction='none'
    x = F.cosine_embedding_loss(input1, input2, target, reduction='none')
    
    # Expand to (batch_size, input_dim) to satisfy VAE input requirements
    x = x.unsqueeze(1).expand(-1, input_dim)
    
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
    
    # This line triggers the bug: AttributeError: 'tuple' object has no attribute 'shape'
    # because output_compile is a tuple (reconstruction, mu, log_var)
    try:
        print(f'Compile  shape: {output_compile.shape}')
    except AttributeError as e:
        print(f"Bug reproduced: {e}")

if __name__ == '__main__':
    main()