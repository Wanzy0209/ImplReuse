import torch

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(9)

# Ensure CUDA is available as the original test case uses cuda
if not torch.cuda.is_available():
    print("Test requires CUDA")
    exit()

def fuzzed_program(arg_0, sentinel):
    var_node_1 = arg_0 # size=(1, 2), stride=(2, 1), dtype=int64, device=cuda
    var_node_5 = torch.full((1, 2), -66, dtype=torch.int32, device='cuda') # size=(1, 2), stride=(2, 1), dtype=int32, device=cuda
    var_node_6 = torch.full((1, 2), 77, dtype=torch.int64, device='cuda') # size=(1, 2), stride=(2, 1), dtype=int64, device=cuda
    var_node_4 = torch.ops.aten.add(var_node_5, var_node_6) # size=(1, 2), stride=(2, 1), dtype=int32, device=cuda
    var_node_7 = torch.full((1, 2), -64, dtype=torch.int32, device='cuda') # size=(1, 2), stride=(2, 1), dtype=int32, device=cuda
    var_node_3 = torch.ops.aten.mul(var_node_4, var_node_7) # size=(1, 2), stride=(2, 1), dtype=int32, device=cuda

    # Adaptation: Replace torch.nonzero with torch.mv
    # torch.mv(input, vec) performs a matrix-vector product.
    # input must be (M, N), vec must be (N). Output is (M).
    # To maintain compatibility with var_node_3 (1, 2) in the subsequent add,
    # we use M=1, N=2.
    mv_mat = torch.randn(1, 2, dtype=torch.float32, device='cuda')
    mv_vec = torch.randn(2, dtype=torch.float32, device='cuda')
    var_node_8 = torch.mv(mv_mat, mv_vec) # size=(1,), dtype=float32

    var_node_2 = torch.ops.aten.add(var_node_3, var_node_8) # size=(1, 2), dtype=float32
    var_node_0 = torch.ops.aten.div(var_node_1, var_node_2) # size=(1, 2), dtype=float32

    # Ensure gradient computation by multiplying with sentinel
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True, device='cuda')

arg_0 = torch.randint(0, 3, (1, 2), dtype=torch.int64, device='cuda')

args = (arg_0,) + (sentinel,)
result_original = fuzzed_program(*args)
print(' eager success')
# Test compilation with unbacked operations - this should work!
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' compile success with torch.mv')

# Compare results - shapes may differ due to data-dependent operations
print(f'Eager result: {result_original}')
print(f'Compiled result: {result_compiled}')
if hasattr(result_original, 'shape') and hasattr(result_compiled, 'shape'):
    print(f'Eager shape: {result_original.shape}')
    print(f'Compiled shape: {result_compiled.shape}')

# Add assertion to verify behavior
assert torch.allclose(result_original, result_compiled), "Divergence detected between eager and compiled modes"