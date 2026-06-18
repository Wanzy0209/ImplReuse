import torch
import torch.nn as nn

class SparseDenseModel(nn.Module):
    """
    Model that converts dense tensor to sparse, performs an operation,
    and converts back to dense.
    """
    def forward(self, x):
        x_sparse = x.to_sparse()
        result = x_sparse * 2
        return result.to_dense()

def test_torch_compile_sparse_operations():
    """
    Test case to verify torch.compile handles sparse tensor conversions
    (to_sparse/to_dense) correctly with the inductor backend.
    """
    # Setup input
    x = torch.randn(10, 10)
    
    model = SparseDenseModel()
    
    # 1. Run in eager mode (baseline)
    eager_output = model(x)
    
    # 2. Run with torch.compile
    # The bug report indicates this fails with NotImplementedError 
    # when using the default 'inductor' backend.
    compiled_model = torch.compile(model)
    compiled_output = compiled_model(x)
    
    # 3. Verify results match
    assert torch.allclose(eager_output, compiled_output), \
        "Compiled output does not match eager output"
    
    print("Test passed: torch.compile works with sparse tensor operations.")

if __name__ == "__main__":
    test_torch_compile_sparse_operations()