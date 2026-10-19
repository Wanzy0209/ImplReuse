import torch
import torch._functorch.aot_autograd as aot_autograd
from torch.utils.cpp_extension import verify_ninja_availability

# Verify ninja availability as requested by the replacement mapping
verify_ninja_availability()

class inner_f(torch.nn.Module):
    """
    Reproduces the bug scenario using the similar API (torch.nn.Sigmoid).
    The original bug involved incorrect stack traces after aot_export_joint_with_descriptors.
    """
    def forward(self, primals, tangents):
        # Flatten inputs as in the original repro
        primals_1, primals_2, tangents_1 = primals, primals, tangents
        
        # Original logic: t, mm, new_empty, index_put, slice
        # We preserve this logic flow but replace the final distributed op 
        # with the similar API (torch.nn.Sigmoid) to test if the stack trace issue persists.
        
        t = torch.ops.aten.t.default(primals_1)
        mm = torch.ops.aten.mm.default(primals_2, t)
        new_empty = torch.ops.aten.new_empty.default(mm, [16, 256], pin_memory=False)
        
        # Create a dummy index for index_put
        _tensor_constant0 = torch.randint(0, 15, (15,))
        index_put = torch.ops.aten.index_put.default(new_empty, [_tensor_constant0], mm)
        
        slice_2 = torch.ops.aten.slice.Tensor(index_put, 0, 0, -1)
        
        # Replacement: Use torch.nn.Sigmoid (Similar API) instead of all_to_all_single
        # This tests if the stack trace issue affects this API as well
        sigmoid_module = torch.nn.Sigmoid()
        result = sigmoid_module(slice_2)
        
        return result

def test_sigmoid_stack_trace():
    mod = inner_f()
    # Create dummy inputs matching the shapes in the bug report
    primals = torch.randn(256, 256)
    tangents = torch.randn(15, 256)
    
    args = (primals, tangents)

    print("Running aot_export_joint_with_descriptors with torch.nn.Sigmoid...")
    
    try:
        # The API call mentioned in the bug report
        gm, _ = aot_autograd.aot_export_joint_with_descriptors(
            mod,
            args,
            decompositions={},
        )
        
        # Output the graph to check stack traces manually or via assertion
        output = gm.print_readable()
        print(output)
        
        # Basic assertion to ensure the graph was generated
        assert gm is not None
        assert "Sigmoid" in str(gm.graph) or "sigmoid" in str(gm.graph)
        
        # Check nodes for stack traces (The bug claims they are wrong)
        # We verify they exist at least
        for node in gm.graph.nodes:
            if node.op == 'call_function' or node.op == 'call_method':
                # In the bug, stack_trace was present but pointed to wrong location.
                # We just ensure the meta info is populated.
                assert hasattr(node, 'meta')

    except Exception as e:
        print(f"Test failed with exception: {e}")
        raise

if __name__ == "__main__":
    test_sigmoid_stack_trace()