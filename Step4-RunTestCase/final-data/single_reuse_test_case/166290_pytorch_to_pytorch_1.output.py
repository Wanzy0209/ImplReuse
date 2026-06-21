import torch
import torch.nn as nn

# Fix: Check if torch._dynamo exists before accessing it to handle older PyTorch versions
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
else:
    print("Warning: torch._dynamo is not available in this PyTorch version. Skipping dynamo config.")

torch.manual_seed(974450504)

# Define a module using torch.nn.ModuleDict, similar to the provided example
class ModuleDictWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict(
            {
                "linear": nn.Linear(1, 10),
            }
        )

    def forward(self, x):
        # Access the layer via the ModuleDict
        return self.layers["linear"](x)

def fuzzed_program(arg_0, arg_1, sentinel):
    var_node_3 = arg_0 # size=(17, 30, 17, 3), stride=(1530, 51, 3, 1), dtype=bool, device=cuda
    var_node_2 = torch.chunk(var_node_3, 3, dim=3)[0] # size=(17, 30, 17, 1), stride=(510, 17, 1, 1), dtype=bool, device=cuda
    var_node_5 = torch.full((17,), 3, dtype=torch.int64) # size=(17,), stride=(1,), dtype=int64, device=cuda
    var_node_6 = arg_1 # size=(15,), stride=(1,), dtype=int64, device=cuda
    _input_size_var_node_4 = var_node_5.size(0)
    _index_var_node_4 = torch.randint(0, _input_size_var_node_4, (15,), device=var_node_5.device)
    var_node_4 = torch.gather(var_node_5, 0, _index_var_node_4) # size=(15,), stride=(1,), dtype=int64, device=cuda
    _input_size_var_node_1 = var_node_2.size(0)
    _index_var_node_1 = torch.randint(0, _input_size_var_node_1, (15,), device=var_node_2.device)
    var_node_1 = torch.index_select(var_node_2, 0, _index_var_node_1) # size=(15, 30, 17, 1), stride=(510, 17, 1, 1), dtype=bool, device=cuda
    
    # --- Adaptation Start ---
    # Original API: var_node_0 = torch.squeeze(var_node_1)
    # Similar API: torch.nn.ModuleDict
    # We adapt the tensor var_node_1 to be compatible with the Linear layer inside ModuleDict.
    # var_node_1 is (15, 30, 17, 1). We flatten it to (N, 1) to pass through Linear(1, 10).
    
    # Convert bool to float for Linear layer
    var_node_1_float = var_node_1.float()
    # Flatten to (7650, 1)
    input_tensor = var_node_1_float.flatten().unsqueeze(-1)
    
    # Instantiate the wrapper containing ModuleDict
    model = ModuleDictWrapper()
    
    # Call the model which uses ModuleDict internally
    var_node_0 = model(input_tensor)
    # --- Adaptation End ---

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

arg_0 = torch.as_strided(torch.randint(0, 2, (26010,), dtype=torch.int8).bool(), (17, 30, 17, 3), (1530, 51, 3, 1))
arg_1 = torch.as_strided(torch.randint(5, 30, (15,)).to(torch.int64), (15,), (1,))

args = (arg_0, arg_1) + (sentinel,)

# Test Eager
result_original = fuzzed_program(*args)
print(' eager success')

# Test Compile
# Fix: Check if torch.compile is available
if hasattr(torch, 'compile'):
    compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(*args)
    print(' compile success')

    # Check for divergence
    assert torch.allclose(result_original, result_compiled), "Divergence detected between eager and compiled outputs"
else:
    print("Skipping torch.compile test as it is not available in this environment.")