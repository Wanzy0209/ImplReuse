import torch
import torch.nn as nn

colab_code = '''
class BoltzmannMachine(nn.Module):
    def __init__(self, num_visible, num_hidden):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_visible, num_hidden) * 0.01)
        self.visible_bias = nn.Parameter(torch.zeros(num_visible))
        self.hidden_bias = nn.Parameter(torch.zeros(num_hidden))
    
    def sample_hidden(self, visible_prob):
        hidden_activations = torch.matmul(visible_prob, self.weights) + self.hidden_bias
        hidden_prob = torch.sigmoid(hidden_activations)
        return torch.bernoulli(hidden_prob), hidden_prob
    
    def free_energy(self, v):
        vbias_term = torch.matmul(v, self.visible_bias)
        wx_b = torch.matmul(v, self.weights) + self.hidden_bias
        hidden_term = torch.sum(torch.log(1 + torch.exp(wx_b)), dim=1)
        return -hidden_term - vbias_term

torch.manual_seed(42)
model = BoltzmannMachine(784, 128)
model.eval()

input_tensor = torch.randint(0, 2, (8, 784), dtype=torch.float32)

# Normal model
with torch.no_grad():
    energy_normal = model.free_energy(input_tensor)
    _, hidden_prob_normal = model.sample_hidden(input_tensor)

# Compiled model
compiled_model = torch.compile(model)
with torch.no_grad():
    energy_compiled = compiled_model.free_energy(input_tensor)
    _, hidden_prob_compiled = compiled_model.sample_hidden(input_tensor)

print(f"Energy difference: {torch.abs(energy_normal - energy_compiled).max():.6f}")
print(f"Hidden prob difference: {torch.abs(hidden_prob_normal - hidden_prob_compiled).max():.6f}")
'''

exec(colab_code)