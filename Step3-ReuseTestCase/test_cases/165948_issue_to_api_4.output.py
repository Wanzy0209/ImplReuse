import torch
from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
import torch.nn.functional as F

class CosineSimilarityModule(torch.nn.Module):
    def __init__(self, dim=1, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1, x2, dim=None, eps=None):
        # Mimic the kwarg handling pattern from the original FlexAttentionModule
        d = dim if dim is not None else self.dim
        e = eps if eps is not None else self.eps
        return F.cosine_similarity(x1, x2, dim=d, eps=e)

# Instantiate the model
cosine_model = CosineSimilarityModule(dim=1, eps=1e-6)

# Setup inputs
batch_size = 2
feature_dim = 16
x1 = torch.randn(batch_size, feature_dim)
x2 = torch.randn(batch_size, feature_dim)

cosine_inputs = (x1, x2)
cosine_kwargs = {"dim": 1, "eps": 1e-6}

# Run eager execution
eager_out = cosine_model(*cosine_inputs, **cosine_kwargs)

# Run export capture using the same pattern as the original bug report
with torch._dynamo.config.patch(install_free_tensors=True):
    gm = _dynamo_graph_capture_for_export(cosine_model)(*cosine_inputs, **cosine_kwargs)

# Verify the exported graph produces the same result
export_out = gm(*cosine_inputs, **cosine_kwargs)
assert torch.allclose(eager_out, export_out), "Exported graph output does not match eager output"