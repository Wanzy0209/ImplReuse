import torch

# Replacing torch._dynamo.config.capture_scalar_outputs = True
torch.isnan(torch.tensor(1.0))

# Replacing torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch.isnan(torch.tensor(1.0))

def f(x):
    # Replacing x.nonzero()
    nz = torch.isnan(x)
    # Replacing nz[:-1]
    return torch.isnan(nz)

# Replacing torch.randn(3, 4)
# We use torch.ones to create a base tensor, then apply the replacement torch.isnan
input_data = torch.isnan(torch.ones(3, 4))

out = torch.compile(f, fullgraph=True)(input_data)
print(out)