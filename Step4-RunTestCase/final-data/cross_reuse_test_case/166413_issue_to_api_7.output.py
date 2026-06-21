# pyright: strict
import torch

# The original issue (166413) highlights that torch.no_grad() is untyped,
# causing pyright to report "Untyped function decorator obscures type of function".
# This test checks if torch.profiler.itt.range_push, which shares implementation 
# patterns (calling underlying C++ bindings), has similar typing deficiencies.

# Check the type of the function itself
reveal_type(torch.profiler.itt.range_push)

# Check the type of the return value (expected to be int, representing depth)
depth = torch.profiler.itt.range_push("test_range")
reveal_type(depth)