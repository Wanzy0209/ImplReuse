import torch

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 4)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(4, 8)
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

device = "cuda"
m = M().to(device)
sample_inputs = (torch.randn(2, 2, device=device),)
eager_out = m(*sample_inputs)

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

compiled_out = loaded_fn(m, *sample_inputs).

assert torch.allclose(eager_out, compiled_out)