import torch
import os

def check_storage_integrity(model, stage_name):
    """
    Leverages torch.is_storage to verify the integrity of model parameters.
    This is relevant to serialization bugs where storage objects might be 
    corrupted or lost during the save/load process.
    """
    for name, param in model.named_parameters():
        # Access the underlying storage of the parameter tensor
        storage = param.data.storage()
        # Use the similar API to check if the object is a valid PyTorch storage
        if not torch.is_storage(storage):
            raise AssertionError(
                f"Storage integrity check failed at stage '{stage_name}' "
                f"for parameter '{name}'. Object is not a valid storage."
            )

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 4)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(4, 8)
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# Use CUDA if available, otherwise CPU to ensure test runs everywhere
device = "cuda" if torch.cuda.is_available() else "cpu"
m = M().to(device)
sample_inputs = (torch.randn(2, 2, device=device),)
eager_out = m(*sample_inputs)

# Check storage integrity before AOT compilation
check_storage_integrity(m, "before_aot_compile")

with torch._dynamo.config.patch("enable_aot_compile", True):
    compiled_fn_path = "./m.pt"
    compiled_fn = torch.compile(
        m,
        fullgraph=True
    ).forward.aot_compile((sample_inputs, {}))

    compiled_fn.save_compiled_function(compiled_fn_path)
    torch._dynamo.reset()
    
    with torch.compiler.set_stance("fail_on_recompile"):
        with open(compiled_fn_path, "rb") as f:
            loaded_fn = torch.compiler.load_compiled_function(f)

assert loaded_fn is not None

# Check storage integrity after loading the compiled function
# The model 'm' is passed to the loaded function, so its state matters
check_storage_integrity(m, "after_load")

# Execute the loaded function
compiled_out = loaded_fn(m, *sample_inputs)

assert torch.allclose(eager_out, compiled_out)

# Cleanup
if os.path.exists(compiled_fn_path):
    os.remove(compiled_fn_path)