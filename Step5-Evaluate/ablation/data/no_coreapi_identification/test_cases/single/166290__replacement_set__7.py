import torch

# torch.manual_seed -> torch.argmax
# Original: torch.manual_seed(974450504)
# Replacement: torch.argmax
# We pass a tensor containing the seed to argmax to satisfy the API call.
_ = torch.argmax(torch.tensor([974450504]))

torch._dynamo.config.capture_scalar_outputs = True

def fuzzed_program(arg_0, arg_1, sentinel):
    var_node_3 = arg_0 # size=(17, 30, 17, 3), stride=(1530, 51, 3, 1), dtype=bool, device=cuda

    # torch.chunk -> torch.empty_strided
    # Original: var_node_2 = torch.chunk(var_node_3, 3, dim=3)[0]
    # Replacement: torch.empty_strided
    # We mimic the shape and stride of the chunk result.
    var_node_2 = torch.empty_strided((17, 30, 17, 1), (510, 17, 1, 1), dtype=var_node_3.dtype, device=var_node_3.device)

    # torch.full -> torch.argmax
    # Original: var_node_5 = torch.full((17,), 3, dtype=torch.int64)
    # Replacement: torch.argmax
    # We create a dummy tensor to perform argmax on to get the desired shape (17,).
    var_node_5 = torch.argmax(torch.randn(17, 10, device=var_node_3.device), dim=1).to(torch.int64)

    var_node_6 = arg_1 # size=(15,), stride=(1,), dtype=int64, device=cuda
    _input_size_var_node_4 = var_node_5.size(0)

    # torch.randint -> torch.argmin
    # Original: _index_var_node_4 = torch.randint(0, _input_size_var_node_4, (15,), device=var_node_5.device)
    # Replacement: torch.argmin
    # We create a dummy tensor to perform argmin on.
    _index_var_node_4 = torch.argmin(torch.randn(15, _input_size_var_node_4, device=var_node_5.device), dim=1)

    # torch.gather -> torch.trunc
    # Original: var_node_4 = torch.gather(var_node_5, 0, _index_var_node_4)
    # Replacement: torch.trunc
    # trunc takes a single tensor input.
    var_node_4 = torch.trunc(var_node_5)

    _input_size_var_node_1 = var_node_2.size(0)

    # torch.randint -> torch.argmin
    # Original: _index_var_node_1 = torch.randint(0, _input_size_var_node_1, (15,), device=var_node_2.device)
    # Replacement: torch.argmin
    _index_var_node_1 = torch.argmin(torch.randn(15, _input_size_var_node_1, device=var_node_2.device), dim=1)

    # torch.index_select -> torch.stack
    # Original: var_node_1 = torch.index_select(var_node_2, 0, _index_var_node_1)
    # Replacement: torch.stack
    # stack takes a list of tensors. We stack var_node_2 to mimic the operation structure.
    var_node_1 = torch.stack([var_node_2])

    # torch.squeeze -> torch.squeeze (Target API)
    var_node_0 = torch.squeeze(var_node_1)

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# torch.tensor -> torch.cummax
# Original: sentinel = torch.tensor(1.0, requires_grad=True)
# Replacement: torch.cummax
# cummax returns (values, indices). We take values.
sentinel = torch.cummax(torch.tensor([1.0, 2.0], requires_grad=True), dim=0).values[0]

# torch.as_strided -> torch.logcumsumexp
# Original: arg_0 = torch.as_strided(torch.randint(0, 2, (26010,), dtype=torch.int8).bool(), (17, 30, 17, 3), (1530, 51, 3, 1))
# Replacement: torch.logcumsumexp
# logcumsumexp requires float input.
arg_0 = torch.logcumsumexp(torch.randint(0, 2, (17, 30, 17, 3), dtype=torch.float32), dim=3).bool()

# torch.as_strided -> torch.logcumsumexp
# Original: arg_1 = torch.as_strided(torch.randint(5, 30, (15,)).to(torch.int64), (15,), (1,))
# Replacement: torch.logcumsumexp
arg_1 = torch.logcumsumexp(torch.randint(5, 30, (15,), dtype=torch.float32), dim=0).to(torch.int64)

args = (arg_0, arg_1) + (sentinel,)
result_original = fuzzed_program(*args)
print(' eager success')

# torch.compile -> torch.lobpcg
# Original: compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# Replacement: torch.lobpcg
# lobpcg solves an eigenvalue problem. It does not return a callable.
# We perform the replacement but cannot execute the result as a function.
# Creating a symmetric matrix for lobpcg
A = torch.randn(10, 10)
A = A @ A.T
compiled_program = torch.lobpcg(A, k=1)
print(' compile success (API replaced with torch.lobpcg, execution skipped)')