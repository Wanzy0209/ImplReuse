import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return (mu, log_var)

    def reparameterize(self, mu, log_var):
        # Replacement: torch.exp -> torch.sigmoid
        std = torch.sigmoid(0.5 * log_var)
        # Replacement: torch.randn_like -> torch.nn.functional.instance_norm
        eps = torch.nn.functional.instance_norm(std)
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
    # Replacement: torch.randn -> torch.nn.functional.adaptive_max_pool2d
    # adaptive_max_pool2d expects 4D input (N, C, H, W). We create a dummy tensor to pool.
    dummy = torch.ones(1, 1, batch_size, input_dim)
    # We pool to the desired dimensions and then reshape to match the model's expected input (Batch, InputDim)
    x = F.adaptive_max_pool2d(dummy, (batch_size, input_dim)).view(batch_size, input_dim)
    return (x,)

def main():
    model = get_default_model()
    model.eval()
    inputs = get_sample_inputs()

    # Eager execution
    with torch.no_grad():
        output_eager = model(*inputs)

    # Verify eager output structure
    assert isinstance(output_eager, tuple), "Eager output should be a tuple"
    assert len(output_eager) == 3, "Eager output should have 3 elements"
    reconstruction_eager, mu_eager, log_var_eager = output_eager
    
    print(f'Eager Reconstruction shape: {reconstruction_eager.shape}')
    print(f'Eager Mu shape: {mu_eager.shape}')
    print(f'Eager Log_var shape: {log_var_eager.shape}')

    # Compiled execution
    compiled_model = torch.compile(model)
    with torch.no_grad():
        output_compile = compiled_model(*inputs)

    # Verify compiled output structure
    # The bug report indicates that accessing .shape on the tuple output causes an error.
    # We verify that the output is indeed a tuple and unpack it to access shapes.
    assert isinstance(output_compile, tuple), "Compiled output should be a tuple"
    assert len(output_compile) == 3, "Compiled output should have 3 elements"

    reconstruction_compile, mu_compile, log_var_compile = output_compile
    print(f'Compile Reconstruction shape: {reconstruction_compile.shape}')
    print(f'Compile Mu shape: {mu_compile.shape}')
    print(f'Compile Log_var shape: {log_var_compile.shape}')

    # Verify shapes match between eager and compiled
    assert reconstruction_eager.shape == reconstruction_compile.shape
    assert mu_eager.shape == mu_compile.shape
    assert log_var_eager.shape == log_var_compile.shape

    print("Test passed successfully!")

if __name__ == '__main__':
    main()