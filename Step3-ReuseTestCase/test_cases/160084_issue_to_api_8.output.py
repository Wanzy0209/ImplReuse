import torch
import torch.nn as nn
import pytest

def test_compile_inductor_with_variable_state_sync():
    """
    Test case for Issue 160084: Regression on compile with backend inductor.
    
    This test preserves the original bug reproduction logic (state mutation in forward)
    while leveraging the pattern from tf.VariableSynchronization by explicitly
    configuring variable types (trainable parameters vs non-trainable buffers)
    to stress the compiler's synchronization handling.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, test requires GPU")

    class RegressionModel(nn.Module):
        def __init__(self, a=0, b=0):
            super().__init__()
            # Trainable parameters
            self.a = nn.Parameter(torch.tensor(a).float())
            self.b = nn.Parameter(torch.tensor(b).float())
            
            # Non-trainable buffer (analogous to tf.Variable with trainable=False)
            # to test synchronization of different variable types.
            self.register_buffer('offset', torch.tensor(1.0))
            
            self.first_batch = True

        def forward(self, x=None):
            # State mutation that causes a graph break
            if self.first_batch:
                # Mimic the logging/check logic from the original bug
                # print(f"Model dtype: {self.a.dtype}. Input dtype: {x.dtype}")
                self.first_batch = False
            
            # Computation involving parameters and buffer
            return x * self.a + self.b + self.offset
   
    model = RegressionModel().cuda()
    
    # Reproduce the exact compilation setup from the bug report
    model.forward = torch.compile(model.forward, backend="inductor")
    
    inputs = torch.randn(4, 10).cuda()
    
    # Execute the compiled model
    # The bug causes: RuntimeError: opt_ready_stream && opt_parent_stream
    try:
        result = model(inputs)
        # Explicit synchronization to catch any async stream errors
        torch.cuda.synchronize()
        
        # Assertions to verify correctness and state change
        assert result is not None
        assert result.shape == (4, 10)
        assert not model.first_batch, "Model state should have mutated"
        
    except RuntimeError as e:
        if "opt_ready_stream && opt_parent_stream" in str(e):
            pytest.fail(f"Regression detected: {e}")
        else:
            raise