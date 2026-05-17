import torch
import torch.nn.functional as F

# Preserve the original bug reproduction logic for input generation
# The original bug involved an int64 tensor created with randint
tensor = torch.randint(low=0, high=10, size=(5,), dtype=torch.int64)

# Adapt the input structure for the similar API (torch.nn.functional.tanh)
# The original bug used [[tensor, 1024], {}] where 1024 was the 'p' argument for mvlgamma.
# tanh only accepts the input tensor, so we adjust the arguments list accordingly.
input_args = [tensor]
input_kwargs = {}

# Leverage the similar API as a candidate for reuse, maintaining the unpacking pattern
# This tests if tanh handles the specific input tensor type and structure correctly
result = F.tanh(*input_args, **input_kwargs)

# Assertions to verify the test case runs successfully and produces expected output
assert result is not None
assert result.shape == tensor.shape
# tanh promotes int64 inputs to float32
assert result.dtype == torch.float32