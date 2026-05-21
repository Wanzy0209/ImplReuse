import torch
import torch.nn as nn

# Enable logging for recompiles as shown in the bug report
torch._logging.set_logs(recompiles=True)

# Define a simple model to mimic the transformer component in the bug report
class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(512, 512)

    def forward(self, x):
        return self.linear(x)

# Instantiate model
model = SimpleTransformer()
# Use bfloat16 as in the original report
model = model.to(dtype=torch.bfloat16)

# Adapt the call site: replace pipe.transformer.compile_repeated_blocks()
# with torch.compile(model)
compiled_model = torch.compile(model)

# Create dummy input
# Shape: (batch_size, sequence_length, hidden_dim)
dummy_input = torch.randn(1, 512, 512, dtype=torch.bfloat16)

# Run inference
# The first run will trigger compilation
print("Running first inference...")
output1 = compiled_model(dummy_input)

# The second run should ideally use the cached compiled graph
# If the bug "region compile triggers recompile" is present, logs will show recompilation here
print("Running second inference...")
output2 = compiled_model(dummy_input)

# Verify outputs are consistent
assert torch.allclose(output1, output2)
print("Test completed.")