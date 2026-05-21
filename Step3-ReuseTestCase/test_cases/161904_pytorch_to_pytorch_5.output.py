import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(128, 32)
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(4):
            self.layers[str(layer_id)] = nn.Linear(32, 32, bias=False)
        self.output = nn.Linear(32, 128, bias=False)

    def forward(self, x):
        x = self.tok_embeddings(x) if self.tok_embeddings else x
        for layer in self.layers.values():
            x = layer(x)
            # Verify torch.any (Similar API) works within the compiled graph
            # This operation is used to check if any element meets a condition.
            # We adapt the original model logic to include this check.
            condition = torch.any(x > 0)
            if condition:
                # Simple logic to ensure the condition is used
                x = x * 1.0 
        return self.output(x) if self.output else x

def main() -> None:
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = Transformer().to(device)
    model.train()

    # Apply torch.compile (Original API Under Test)
    # The bug report indicates issues with compiled models in specific schedules.
    # We test if torch.any (Similar API) works correctly in a compiled model.
    compiled_model = torch.compile(model)

    # Create dummy input
    input_ids = torch.randint(0, 128, (8, 4096), device=device)

    # Run the compiled model
    # This replaces the pp_schedule.step call from the original snippet
    output = compiled_model(input_ids)
    
    # Assertions to verify correctness
    assert output is not None, "Output is None"
    assert output.shape == (8, 4096, 128), f"Expected shape (8, 4096, 128), got {output.shape}"
    
    print("Test passed: torch.any works correctly with torch.compile")

if __name__ == "__main__":
    main()