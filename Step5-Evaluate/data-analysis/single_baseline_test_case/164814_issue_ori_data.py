# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(1000560)

def fuzzed_program(sentinel):
    var_node_3 = torch.full((2, 3), 3, dtype=torch.int32) # size=(2, 3), stride=(3, 1), dtype=int32, device=cuda
    _inp_unique_wide = torch.arange(1, device=var_node_3.device, dtype=torch.int64)
    _uniq_wide = torch.unique(_inp_unique_wide)
    var_node_2 = _uniq_wide.to(var_node_3.dtype) # size=(1,), stride=(1,), dtype=int32, device=cuda
    var_node_1 = torch.reshape(var_node_2, [1]) # size=(1,), stride=(1,), dtype=int32, device=cuda
    var_node_0 = torch.squeeze(var_node_1) # size=(), stride=(), dtype=int32, device=cuda
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)


args = () + (sentinel,)
result_original = fuzzed_program(*args)
print('✅ eager success')
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print('✅ compile success')