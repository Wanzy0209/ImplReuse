import torch
import torch.nn.functional as F
from torch.profiler import itt

def test_tanh_with_profiler_range():
    """
    Test case adapted from Issue 164143 logic.
    Original: DebugMode silently disables torch.compile.
    Adaptation: Verifies torch.nn.functional.tanh execution 
    within torch.profiler.itt.range_pop context.
    """
    # Setup input tensor
    x = torch.randn(2, 2)
    
    # Original logic: Setup context (DebugMode) -> Compile -> Run
    # Adapted logic: Setup profiler range -> Run function -> End range
    
    # Start a profiling range (Context setup)
    itt.range_push("tanh_range")
    
    # Call the Similar API: torch.nn.functional.tanh
    # This replaces the 'forward' function in the original bug report
    y = F.tanh(x)
    
    # Replacement API: torch.profiler.itt.range_pop
    # This replaces the interaction of torch.compile and DebugMode
    itt.range_pop()
    
    # Assertion to ensure the operation ran correctly
    expected = torch.tanh(x)
    assert torch.allclose(y, expected)

if __name__ == "__main__":
    test_tanh_with_profiler_range()