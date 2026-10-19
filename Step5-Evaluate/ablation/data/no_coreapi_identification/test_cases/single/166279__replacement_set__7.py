import torch
import torch.nn.functional as F

# Replacement: torch._dynamo.config.capture_scalar_outputs = True -> torch.nn.functional.sigmoid
F.sigmoid(torch.tensor(1.0))

# Replacement: torch.manual_seed(1166094474) -> torch.argmax
torch.argmax(torch.tensor([1166094474]))

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, sentinel):
    # Replacement: torch.full((12,), False, dtype=torch.bool) -> torch.argmax
    # Original: size=(12,)
    # New: argmax on (12, 5) -> (12,)
    var_node_3 = torch.argmax(torch.randn(12, 5), dim=1)

    # Replacement: torch.chunk(var_node_3, 4, dim=0)[0] -> torch.nn.functional.sigmoid
    # Original: size=(3,)
    # New: sigmoid on (12,) -> (12,)
    var_node_2 = F.sigmoid(var_node_3)

    var_node_6 = arg_0
    var_node_7 = arg_1
    _input_size_var_node_5 = var_node_6.size(0)

    # Replacement: torch.randint(0, _input_size_var_node_5, (10,), device=var_node_6.device) -> torch.argmin
    # Original: size=(10,)
    # New: argmin on (10, 5) -> (10,)
    _index_var_node_5 = torch.argmin(torch.randn(10, 5), dim=1)

    # Replacement: torch.gather(var_node_6, 0, _index_var_node_5) -> torch.trunc
    # Original: size=(10,)
    # New: trunc on (12,) -> (12,)
    var_node_5 = torch.trunc(var_node_6)

    # Replacement: torch.chunk(var_node_5, 2, dim=0)[0] -> torch.nn.functional.sigmoid
    # Original: size=(5,)
    # New: sigmoid on (12,) -> (12,)
    var_node_4 = F.sigmoid(var_node_5)

    var_node_10 = arg_2

    # Replacement: torch.chunk(var_node_10, 4, dim=1)[0] -> torch.nn.functional.sigmoid
    # Original: size=(6, 1)
    # New: sigmoid on (6, 4) -> (6, 4)
    var_node_9 = F.sigmoid(var_node_10)

    # Replacement: torch.squeeze(var_node_9) -> torch.distributions.constraints.cat
    # Original: size=(6,)
    # New: cat on (6, 4) -> (6, 4) (using torch.cat as proxy for runnability)
    var_node_8 = torch.cat([var_node_9], dim=0)

    # Replacement: torch.cat([var_node_2, var_node_4, var_node_8], dim=0) -> torch.isnan
    # Original: size=(14,)
    # New: isnan on stack of (12,), (12,), (6, 4) -> reshaped to (12,) for compatibility
    var_node_1 = torch.isnan(torch.stack([var_node_2, var_node_4, var_node_8.reshape(12)]))

    var_node_11 = arg_3

    # Replacement: torch.cat([var_node_1, var_node_11], dim=0) -> torch.isnan
    # Original: size=(16,)
    # New: isnan on stack of (3, 12) and (2,) -> reshaped to (2,) for compatibility
    var_node_0 = torch.isnan(torch.stack([var_node_1.reshape(2), var_node_11]))

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Replacement: torch.tensor(1.0, requires_grad=True) -> torch.cummax
# Original: scalar
# New: cummax on [1.0] -> values (1,)
sentinel = torch.cummax(torch.tensor([1.0]), dim=0)[0]
sentinel.requires_grad = True

# Replacement: torch.as_strided(...) -> torch.logcumsumexp
# arg_0: (12,)
arg_0 = torch.logcumsumexp(torch.randn(12), dim=0)
# arg_1: (10,)
arg_1 = torch.logcumsumexp(torch.randn(10), dim=0)
# arg_2: (6, 4)
arg_2 = torch.logcumsumexp(torch.randn(6, 4), dim=0)
# arg_3: (2,)
arg_3 = torch.logcumsumexp(torch.randn(2), dim=0)

args = (arg_0, arg_1, arg_2, arg_3) + (sentinel,)
result_original = fuzzed_program(*args)
print(' eager success')

# Replacement: torch.compile(...) -> torch.lobpcg
# Original: compiled_program = torch.compile(...)
# New: Just call lobpcg
# The original code calls compiled_program(*args).
# I will replace the call with lobpcg.
# lobpcg(A, k). A must be square.
# I'll create a dummy A.
A = torch.randn(10, 10)
result_compiled = torch.lobpcg(A, k=1)
print(' compile success')