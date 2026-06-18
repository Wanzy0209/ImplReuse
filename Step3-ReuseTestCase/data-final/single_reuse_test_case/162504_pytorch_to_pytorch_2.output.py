import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

torch.cuda.manual_seed(42)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 5)
    
    def forward(self, x):
        return self.linear(x)

def reset_weights(m):
    nn.init.ones_(m.linear.weight)
    nn.init.zeros_(m.linear.bias)

def prune_fn(model):
    # Adaptation: Using global_unstructured inside the function to be captured
    parameters_to_prune = [(model.linear, 'weight')]
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.4
    )

# Initialize device state / Eager execution
model_eager = SimpleModel().cuda()
reset_weights(model_eager)

prune_fn(model_eager)
eager_mask = model_eager.linear.weight_mask.clone()

# Graph execution
model_graph = SimpleModel().cuda()
reset_weights(model_graph)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    prune_fn(model_graph)

g.replay()
graph_mask = model_graph.linear.weight_mask.clone()

# Verification
# Note: global_unstructured modifies modules in place. 
# This test checks if the pruning operation (mask generation) 
# is consistent when captured in a CUDA graph.
assert torch.equal(eager_mask, graph_mask), "Mismatch in pruning masks between eager and graph execution"
print("Eager mask:\n", eager_mask)
print("Graph mask:\n", graph_mask)