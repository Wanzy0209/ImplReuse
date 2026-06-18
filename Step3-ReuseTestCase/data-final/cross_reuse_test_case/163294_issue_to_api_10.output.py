import torch

class LogSumExpNoGradCase(torch.nn.Module):
    def forward(self, x):
        with torch.no_grad():
            # Leveraging the similar API torch.special.logsumexp inside the no_grad context
            y = torch.special.logsumexp(x, dim=0)
        return y

# Initial export
ep = torch.export.export(
    LogSumExpNoGradCase(),
    (torch.randn(6),),
    strict=False,
)

# Retracing the exported module
# This is the step that triggers the bug where the submod becomes empty
ep2 = torch.export.export(ep.module(), (torch.randn(6),))

# Verification logic
# The bug report indicates that 'submod_1' (the body of no_grad) becomes empty.
# We check if the graph contains the operation from the similar API.
submod_name = "submod_1"
if hasattr(ep2.graph_module, submod_name):
    submod = getattr(ep2.graph_module, submod_name)
    graph_str = str(submod.graph)
    print(f"Graph of {submod_name}:\n{graph_str}")
    
    # Assert that the submod is not empty and contains the expected operation
    assert "logsumexp" in graph_str, "Bug reproduced: submod is empty or missing the logsumexp operation."
else:
    # Fallback if naming convention differs, print the whole program
    print(ep2)