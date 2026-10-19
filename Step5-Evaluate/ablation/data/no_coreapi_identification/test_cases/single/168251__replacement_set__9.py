import tensorflow as tf
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
        # Replacement for torch.randn_like: torch.nn.Conv2d
        # We use a 1x1 convolution to generate noise of the same shape
        self.noise_gen = torch.nn.Conv2d(latent_dim, latent_dim, kernel_size=1)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return (mu, log_var)

    def reparameterize(self, mu, log_var):
        # Original: std = torch.exp(0.5 * log_var)
        # Replacement request: torch.exp -> tf.raw_ops.HashTable
        # Note: tf.raw_ops.HashTable is a TensorFlow API and cannot be used in PyTorch.
        # Keeping torch.exp to maintain runnability in PyTorch.
        std = torch.exp(0.5 * log_var)
        
        # Original: eps = torch.randn_like(std)
        # Replacement: torch.nn.Conv2d
        # We adapt the logic to use Conv2d to generate a tensor of the same shape as std
        # Reshape std to (Batch, Channels, Height, Width) for Conv2d
        std_4d = std.unsqueeze(-1).unsqueeze(-1) 
        eps_4d = self.noise_gen(std_4d)
        eps = eps_4d.squeeze(-1).squeeze(-1)
        
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
    # Original: x = torch.randn(batch_size, input_dim)
    # Replacement: torch.nn.functional.hinge_embedding_loss
    # We use hinge_embedding_loss with reduction='none' to generate a tensor of the correct shape
    # instead of random numbers.
    dummy_input = torch.ones(batch_size, input_dim)
    dummy_target = torch.ones(batch_size, input_dim) * -1
    x = torch.nn.functional.hinge_embedding_loss(dummy_input, dummy_target, reduction='none')
    return (x,)

def main():
    print("Testing torch.compile with tuple output...")
    model = get_default_model()
    model.eval()
    inputs = get_sample_inputs()
    
    # 1. Test Eager Mode
    print("\n1. Testing Eager Mode:")
    with torch.no_grad():
        (reconstruction, mu, log_var) = model(*inputs)
    print(f'   Input shape: {inputs[0].shape}')
    print(f'   Reconstruction shape: {reconstruction.shape}')
    print(f'   Mu shape: {mu.shape}')
    print(f'   Log_var shape: {log_var.shape}')
    print("   Eager mode executed successfully (tuple unpacking works).")

    # 2. Test Compiled Mode
    print("\n2. Testing Compiled Mode (torch.compile):")
    compiled_model = torch.compile(model)
    with torch.no_grad():
        output_compile = compiled_model(*inputs)
    
    print(f'   Output type: {type(output_compile)}')
    
    # 3. Verify the bug behavior
    # The bug report states that accessing .shape on the tuple output triggers AttributeError.
    # We expect this to fail because the model returns a tuple, not a single tensor.
    print("\n3. Verifying bug behavior (accessing .shape on tuple):")
    try:
        # This line should trigger AttributeError: 'tuple' object has no attribute 'shape'
        _ = output_compile.shape
        print("   ERROR: Expected AttributeError but code succeeded.")
    except AttributeError as e:
        print(f"   Caught expected AttributeError: {e}")
        print("   This confirms the behavior described in the bug report.")
        
    # 4. Verify correct usage (unpacking)
    print("\n4. Verifying correct usage (unpacking compiled output):")
    try:
        (rec_c, mu_c, log_var_c) = output_compile
        print(f"   Unpacked successfully. Shapes: {rec_c.shape}, {mu_c.shape}, {log_var_c.shape}")
    except Exception as e:
        print(f"   ERROR: Failed to unpack tuple: {e}")

if __name__ == '__main__':
    main()