import torch
import torch.nn as nn
import torch.fx as fx

# Similar API: torch.nn.Sigmoid
# Replacements: 
# - torch.ops.aten.t.default -> torch.eig
# - torch.ops.aten.mm.default -> torch.eig
# - torch.ops.aten.new_empty.default -> torch.eig
# - torch.ops.aten.index_put.default -> torch.eig
# - torch.ops.aten.slice.Tensor -> torch.eig
# - torch.ops._c10d_functional.all_to_all_single.default -> torch.eig
# - torch.ops._c10d_functional.wait_tensor.default -> torch.eig
# - torch._functorch.aot_autograd.aot_export_joint_with_descriptors -> torch.eig

class ReproModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Leveraging the similar API: torch.nn.Sigmoid
        self.sigmoid = nn.Sigmoid()
        # Mocking the tensor constant from the original test case
        self.register_buffer("_tensor_constant0", torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]))

    def forward(self, primals, tangents):
        # Adapt inputs for torch.eig (requires square matrix)
        # We assume primals is a square tensor for this modified test case
        primals_1 = primals
        primals_2 = primals[:, :15] # Slice to match original dimension logic if needed, but eig needs square.
        
        # To make torch.eig runnable, we ensure inputs are square.
        # Original: t: "f32[256, 256]" = torch.ops.aten.t.default(primals_1)
        # Replacement: torch.eig
        # Note: torch.eig returns (eigenvalues, eigenvectors)
        t_vals, t_vecs = torch.eig(primals_1)
        
        # Original: mm: "f32[15, 256]" = torch.ops.aten.mm.default(primals_2, t)
        # Replacement: torch.eig
        # We use a square subset of primals_2 for eig
        square_input = primals_1[:15, :15]
        mm_vals, mm_vecs = torch.eig(square_input)
        
        # Original: new_empty: "f32[16, 256]" = torch.ops.aten.new_empty.default(mm, [16, 256], pin_memory = False)
        # Replacement: torch.eig
        # We use a 16x16 subset
        new_empty_input = primals_1[:16, :16]
        new_vals, new_vecs = torch.eig(new_empty_input)
        
        # Original: index_put: "f32[16, 256]" = torch.ops.aten.index_put.default(new_empty, [_tensor_constant0], mm)
        # Replacement: torch.eig
        # We reuse the square input
        index_vals, index_vecs = torch.eig(new_empty_input)
        
        # Original: slice_2: "f32[15, 256]" = torch.ops.aten.slice.Tensor(index_put, 0, 0, -1)
        # Replacement: torch.eig
        slice_input = primals_1[:15, :15]
        slice_vals, slice_vecs = torch.eig(slice_input)
        
        # Original: all_to_all_single: "f32[15, 256]" = torch.ops._c10d_functional.all_to_all_single.default(...)
        # Replacement: torch.eig
        all_vals, all_vecs = torch.eig(slice_input)
        
        # Original: wait_tensor: "f32[15, 256]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single)
        # Replacement: torch.eig
        wait_vals, wait_vecs = torch.eig(slice_input)
        
        # Apply the similar API (Sigmoid) to the result
        result = self.sigmoid(wait_vals)
        
        return result

def test_repro():
    # Create a square matrix input for torch.eig
    # Original dimensions were 256, 256
    input_tensor = torch.randn(256, 256, dtype=torch.float32)
    tangents_tensor = torch.randn(256, 256, dtype=torch.float32)
    
    model = ReproModule()
    
    # Original API: torch._functorch.aot_autograd.aot_export_joint_with_descriptors
    # Replacement API: torch.eig
    # Instead of exporting, we call torch.eig on the model output or input to satisfy the replacement.
    # Here we run the model (which contains torch.eig calls) and then call torch.eig on the result.
    
    model_output = model(input_tensor, tangents_tensor)
    
    # Final replacement of the export mechanism with torch.eig
    # This simulates the "test" logic being replaced.
    final_eig_vals, final_eig_vecs = torch.eig(model_output)
    
    # Assertions to ensure it runs
    assert final_eig_vals is not None
    assert final_eig_vals.shape[0] == 256 # Sigmoid output preserves shape, eig preserves shape
    
    print("Test case executed successfully with replacements.")

if __name__ == "__main__":
    test_repro()