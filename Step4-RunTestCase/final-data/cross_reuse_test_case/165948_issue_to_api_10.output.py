import torch

# Handle the missing module gracefully by checking availability
try:
    from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
    HAS_DYNAMO = True
except (ImportError, ModuleNotFoundError):
    HAS_DYNAMO = False

# Translating the logic from tf.nn.collapse_repeated to PyTorch
# based on the provided similar API information.
def collapse_repeated_mask(labels, seq_length, name=None):
    """
    Mimics the mask creation logic found in tf.nn.collapse_repeated.
    Original TF logic:
        label_mask = array_ops.concat([
            array_ops.ones_like(labels[:, :1], dtypes.bool),
            math_ops.not_equal(labels[:, 1:], labels[:, :-1])
        ], axis=1)
    """
    # Mask labels that don't equal previous label.
    # First element is always kept (True)
    first_mask = torch.ones_like(labels[:, :1], dtype=torch.bool)
    
    # Subsequent elements are kept if they differ from the previous one
    rest_mask = torch.ne(labels[:, 1:], labels[:, :-1])
    
    # Concatenate the masks
    label_mask = torch.cat([first_mask, rest_mask], dim=1)
    
    return label_mask

class CollapseRepeatedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, labels, seq_length, name=None):
        # We reuse the translated logic here, passing 'name' as a kwarg
        # to mirror the structure of the original bug report (block_mask kwarg).
        return collapse_repeated_mask(labels, seq_length, name=name)

# Setup test data
batch_size = 2
max_seq_len = 10
labels = torch.randint(0, 5, (batch_size, max_seq_len))
seq_length = torch.tensor([max_seq_len, max_seq_len - 2])

model = CollapseRepeatedModule()

# Define inputs and kwargs
# The original bug used flex_kwargs = {"block_mask": block_mask}
# Here we use a kwarg 'name' to test kwarg handling with the similar API logic
inputs = (labels, seq_length)
kwargs = {"name": "test_collapse"}

# Run eager execution
eager_out = model(*inputs, **kwargs)

# Run with torch._dynamo if available
if HAS_DYNAMO:
    # This attempts to reproduce the tracing issue using the similar API's pattern
    with torch._dynamo.config.patch(install_free_tensors=True):
        gm = _dynamo_graph_capture_for_export(model)(*inputs, **kwargs)

    # Verify the output matches
    dynamo_out = gm(*inputs, **kwargs)
    assert torch.equal(eager_out, dynamo_out), "Eager and Dynamo outputs do not match"
else:
    print("Skipping Dynamo test: torch._dynamo module not found.")