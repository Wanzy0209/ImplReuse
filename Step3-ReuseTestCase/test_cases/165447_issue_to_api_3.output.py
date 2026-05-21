import torch
import tempfile
import os

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 4)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(4, 8)
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

def test_aot_precompile_multiple_runs():
    # Ensure CUDA is available for the test
    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available")
        return

    # Leverage the similar API to check backend capability/state
    # This ensures the CUDA context is queried before AOT operations
    is_fa_available = torch.backends.cuda.is_flash_attention_available()
    print(f"Flash Attention available: {is_fa_available}")

    device = "cuda"
    m = M().to(device)
    sample_inputs = (torch.randn(2, 2, device=device),)
    eager_out = m(*sample_inputs)

    # Reproduce the "running multiple times" failure scenario
    # by iterating the serialization/deserialization process
    for i in range(2):
        print(f"Running iteration {i+1}...")
        
        with torch._dynamo.config.patch("enable_aot_compile", True):
            # Use a temporary file for safe cleanup
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                compiled_fn_path = tmp.name
            
            try:
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

                # Execute the loaded function
                compiled_out = loaded_fn(m, *sample_inputs)

                # Verify correctness
                assert torch.allclose(eager_out, compiled_out)
            finally:
                # Clean up the serialized file
                if os.path.exists(compiled_fn_path):
                    os.remove(compiled_fn_path)

if __name__ == "__main__":
    test_aot_precompile_multiple_runs()