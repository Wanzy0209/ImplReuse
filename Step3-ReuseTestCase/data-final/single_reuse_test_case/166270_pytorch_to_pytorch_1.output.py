import torch
import torch.nn as nn

torch._dynamo.config.capture_scalar_outputs = True
torch.manual_seed(1061983224)

class ModuleDictModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Using ModuleDict as the similar API to test
        self.layers = nn.ModuleDict({
            "0": nn.Identity(), # Identity used to handle bool tensor inputs
        })

    def forward(self, x):
        # Replicate the tensor manipulation sequence from the original bug report
        # to test if ModuleDict interacts correctly with these specific tensor states.
        var_node_4 = x
        var_node_3 = torch.chunk(var_node_4, 4, dim=0)[0]
        var_node_2 = torch.squeeze(var_node_3)
        var_node_1 = torch.stack([var_node_2], dim=0)
        var_node_0 = torch.reshape(var_node_1, [1])
        
        # Pass the result through the ModuleDict
        result = self.layers["0"](var_node_0)
        return result

# Input generation from the original bug report
arg_0 = torch.as_strided(torch.randint(0, 2, (4,), dtype=torch.int8).bool(), (4,), (1,))

model = ModuleDictModel()

# Test Eager execution
print('Testing Eager...')
try:
    result_original = model(arg_0)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')

# Test Compiled execution
print('Testing Compile...')
try:
    compiled_model = torch.compile(model, fullgraph=True, dynamic=True)
    result_compiled = compiled_model(arg_0)
    print(' compile success')
except Exception as e:
    print(f' compile failed: {e}')