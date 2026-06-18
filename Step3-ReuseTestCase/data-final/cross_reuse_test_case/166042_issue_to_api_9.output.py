import torch
import torch.nn.functional as F

def test_cudnn_sdp_enabled_with_dynamo_matmul():
    """
    Test case for torch.backends.cuda.cudnn_sdp_enabled.
    This test preserves the logic from the original bug report (Issue 166042),
    which involved a complex chain of bfloat16 matmuls on CUDA under torch._dynamo.
    The original bug triggered an assertion in torch.nn.functional.embedding.
    Here we verify that the similar API (cudnn_sdp_enabled) functions correctly
    within the same execution context.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    # Configuration from the original bug report
    torch._dynamo.config.capture_scalar_outputs = True
    torch.manual_seed(1352030645)

    # Initialize arguments based on the comments in the original fuzzed_program
    # arg_0: size=(4, 8), stride=(8, 1), dtype=bfloat16, device=cuda
    arg_0 = torch.randn(4, 8, dtype=torch.bfloat16, device='cuda')
    # arg_1: size=(7, 12), stride=(12, 1), dtype=bfloat16, device=cuda
    arg_1 = torch.randn(7, 12, dtype=torch.bfloat16, device='cuda')
    # arg_2: size=(12, 2), stride=(2, 1), dtype=bfloat16, device=cuda
    arg_2 = torch.randn(12, 2, dtype=torch.bfloat16, device='cuda')
    # arg_3: size=(14, 9), stride=(9, 1), dtype=bfloat16, device=cuda
    arg_3 = torch.randn(14, 9, dtype=torch.bfloat16, device='cuda')
    # arg_4: size=(2,), stride=(1,), dtype=bfloat16, device=cuda
    arg_4 = torch.randn(2, dtype=torch.bfloat16, device='cuda')
    # arg_5: size=(478, 13) - truncated in original, creating placeholder
    arg_5 = torch.randn(478, 13, dtype=torch.bfloat16, device='cuda')
    
    # Unused sentinel from original signature
    sentinel = None

    def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, sentinel):
        # Leverage the similar API: torch.backends.cuda.cudnn_sdp_enabled
        # We check this flag to ensure it interacts correctly with the dynamo graph
        # and the bfloat16 matmul operations.
        sdp_enabled = torch.backends.cuda.cudnn_sdp_enabled()

        # Reproduce the matmul logic from the original bug report
        var_node_4 = arg_0
        var_node_5 = torch.full((8, 7), -0.80078125, dtype=torch.bfloat16, device='cuda')
        var_node_3 = torch.matmul(var_node_4.to(torch.bfloat16), var_node_5.to(torch.bfloat16))
        
        var_node_7 = arg_1
        var_node_8 = arg_2
        var_node_6 = torch.matmul(var_node_7.to(torch.bfloat16), var_node_8.to(torch.bfloat16))
        
        var_node_2 = torch.matmul(var_node_3.to(torch.bfloat16), var_node_6.to(torch.bfloat16))
        
        var_node_11 = torch.full((2, 3), 1.515625, dtype=torch.bfloat16, device='cuda')
        var_node_12 = torch.full((3, 16), 0.2353515625, dtype=torch.bfloat16, device='cuda')
        var_node_10 = torch.matmul(var_node_11.to(torch.bfloat16), var_node_12.to(torch.bfloat16))
        
        var_node_14 = torch.full((16, 4), 2.21875, dtype=torch.bfloat16, device='cuda')
        var_node_15 = torch.full((4, 9), -1.7421875, dtype=torch.bfloat16, device='cuda')
        var_node_13 = torch.matmul(var_node_14.to(torch.bfloat16), var_node_15.to(torch.bfloat16))
        
        var_node_9 = torch.matmul(var_node_10.to(torch.bfloat16), var_node_13.to(torch.bfloat16))
        var_node_1 = torch.matmul(var_node_2.to(torch.bfloat16), var_node_9.to(torch.bfloat16))
        
        var_node_19 = arg_3
        var_node_20 = torch.full((9, 2), 0.8203125, dtype=torch.bfloat16, device='cuda')
        var_node_18 = torch.matmul(var_node_19.to(torch.bfloat16), var_node_20.to(torch.bfloat16))
        
        var_node_22 = arg_4
        var_node_23 = torch.full((2,), -0.7421875, dtype=torch.bfloat16, device='cuda')
        var_node_21 = torch.add(var_node_22, var_node_23)
        var_node_17 = torch.matmul(var_node_18.to(torch.bfloat16), var_node_21.to(torch.bfloat16))
        
        # The original test case continues here, but we stop at the truncation point.
        # We return the result of the similar API check and the last computed tensor.
        return sdp_enabled, var_node_17

    # Compile the program with torch._dynamo
    compiled_program = torch.compile(fuzzed_program)
    
    # Execute with dummy arguments for unused slots
    result = compiled_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, None, None, sentinel)
    
    # Assertions
    # 1. Check that cudnn_sdp_enabled returns a boolean
    assert isinstance(result[0], bool), "cudnn_sdp_enabled should return a boolean"
    
    # 2. Check that the matmul operations produced a valid tensor
    assert isinstance(result[1], torch.Tensor), "Matmul operations should return a Tensor"
    assert result[1].dtype == torch.bfloat16, "Output dtype should be bfloat16"
    assert result[1].device.type == 'cuda', "Output device should be cuda"
    
    print("Test passed.")

if __name__ == "__main__":
    test_cudnn_sdp_enabled_with_dynamo_matmul()