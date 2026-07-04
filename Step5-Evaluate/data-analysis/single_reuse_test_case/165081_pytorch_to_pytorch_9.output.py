import torch

# Check for torch._dynamo availability
try:
    import torch._dynamo
except ModuleNotFoundError:
    print("torch._dynamo is not available. Skipping test.")
    exit(0)

# Configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

# Set the seed for reproducibility
torch.manual_seed(52676)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, *args):
    # Setup device based on inputs
    device = arg_0.device
    
    var_node_6 = arg_0
    var_node_7 = arg_1
    var_node_5 = torch.matmul(var_node_6.to(torch.float64), var_node_7.to(torch.float64))
    
    var_node_9 = torch.full((9, 11, 12), 1.5758497316910556, dtype=torch.float64, device=device)
    var_node_10 = arg_2
    var_node_8 = torch.matmul(var_node_9.to(torch.float64), var_node_10.to(torch.float64))
    
    var_node_4 = torch.matmul(var_node_5.to(torch.float64), var_node_8.to(torch.float64))
    
    var_node_13 = arg_3
    var_node_14 = arg_4
    var_node_12 = torch.matmul(var_node_13.to(torch.float64), var_node_14.to(torch.float64))
    
    var_node_15 = arg_5
    var_node_11 = torch.matmul(var_node_12.to(torch.float64), var_node_15.to(torch.float64))
    
    var_node_3 = torch.matmul(var_node_4.to(torch.float64), var_node_11.to(torch.float64))
    
    var_node_17 = arg_6
    var_node_18 = arg_7
    var_node_16 = torch.matmul(var_node_17.to(torch.float64), var_node_18.to(torch.float64))
    
    var_node_2 = torch.matmul(var_node_3.to(torch.float64), var_node_16.to(torch.float64))
    
    var_node_23 = torch.full((156, 8), -0.5249394453404403, dtype=torch.float64, device=device)
    var_node_24 = torch.full((8, 9), 0.9331226188585692, dtype=torch.float64, device=device)
    var_node_22 = torch.matmul(var_node_23.to(torch.float64), var_node_24.to(torch.float64))
    
    var_node_26 = torch.full((9, 13), -0.9276381954691514, dtype=torch.float64, device=device)

    # Adaptation: Replace torch.nonzero with torch.argmax
    # We apply it to var_node_2 which is a result of data-dependent operations (matmul)
    # to test the behavior in a similar context.
    return torch.argmax(var_node_2, dim=1)

def main():
    # Check for CUDA availability as the original bug report used CUDA
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return

    device = torch.device("cuda")

    # Initialize arguments based on the comments in the original test case
    # arg_0: size=(9, 9, 9), stride=(81, 9, 1), dtype=float64, device=cuda
    arg_0 = torch.randn(9, 9, 9, dtype=torch.float64, device=device)
    # arg_1: size=(9, 9, 11), stride=(99, 11, 1), dtype=float64, device=cuda
    arg_1 = torch.randn(9, 9, 11, dtype=torch.float64, device=device)
    # arg_2: size=(9, 12, 8), stride=(96, 8, 1), dtype=float64, device=cuda
    arg_2 = torch.randn(9, 12, 8, dtype=torch.float64, device=device)
    # arg_3: size=(9, 8, 13), stride=(104, 13, 1), dtype=float64, device=cuda
    arg_3 = torch.randn(9, 8, 13, dtype=torch.float64, device=device)
    # arg_4: size=(9, 13, 7), stride=(91, 7, 1), dtype=float64, device=cuda
    arg_4 = torch.randn(9, 13, 7, dtype=torch.float64, device=device)
    # arg_5: size=(9, 7, 16), stride=(112, 16, 1), dtype=float64, device=cuda
    arg_5 = torch.randn(9, 7, 16, dtype=torch.float64, device=device)
    # arg_6: size=(9, 16, 12), stride=(192, 12, 1), dtype=float64, device=cuda
    arg_6 = torch.randn(9, 16, 12, dtype=torch.float64, device=device)
    # arg_7: size=(9, 12, 11), stride=(132, 11, 1), dtype=float64, device=cuda
    arg_7 = torch.randn(9, 12, 11, dtype=torch.float64, device=device)

    # Run eager mode
    try:
        eager_result = fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7)
        print("Eager execution successful.")
    except Exception as e:
        print(f"Eager execution failed: {e}")
        return

    # Run compiled mode
    try:
        compiled_program = torch._dynamo.optimize()(fuzzed_program)
        compiled_result = compiled_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7)
        print("Compiled execution successful.")
    except Exception as e:
        print(f"Compiled execution failed: {e}")
        return

    # Check for divergence
    if torch.equal(eager_result, compiled_result):
        print("Test Passed: Eager and Compiled results match.")
    else:
        print("Test Failed: Eager and Compiled results diverge.")
        print(f"Eager result: {eager_result}")
        print(f"Compiled result: {compiled_result}")

if __name__ == "__main__":
    main()