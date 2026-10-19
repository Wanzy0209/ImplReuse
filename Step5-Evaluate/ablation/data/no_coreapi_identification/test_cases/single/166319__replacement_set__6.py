import tensorflow as tf
import torch

# Configuration from the original test case
torch._dynamo.config.capture_scalar_outputs = True

# Replacement: torch.manual_seed -> torch.isnan
# Original: torch.manual_seed(1215252001)
# Replacement: torch.isnan(torch.tensor(1215252001))
torch.isnan(torch.tensor(1215252001))

def fuzzed_program(arg_0, arg_1, arg_2, sentinel):
    var_node_4 = arg_0 # size=(9, 1, 15, 4), stride=(60, 60, 0, 1), dtype=int32, device=cuda
    
    # Replacement: torch.squeeze -> torch.as_strided
    # Original: var_node_3 = torch.squeeze(var_node_4) # size=(9, 15, 4)
    # Replacement: var_node_3 = torch.as_strided(var_node_4, (9, 15, 4), (60, 4, 1))
    var_node_3 = torch.as_strided(var_node_4, (9, 15, 4), (60, 4, 1))
    
    # Replacement: torch.chunk -> torch.triu
    # Original: var_node_2 = torch.chunk(var_node_3, 4, dim=2)[0] # size=(9, 15, 1)
    # Replacement: var_node_2 = torch.triu(var_node_3) # size=(9, 15, 4)
    var_node_2 = torch.triu(var_node_3)
    
    # Replacement: torch.squeeze -> torch.as_strided
    # Original: var_node_1 = torch.squeeze(var_node_2) # size=(9, 15)
    # Replacement: var_node_1 = torch.as_strided(var_node_2, (9, 15, 4), (60, 4, 1))
    # Note: Original var_node_2 was (9, 15, 1), so squeeze gave (9, 15).
    # My replacement var_node_2 is (9, 15, 4), so "squeeze" (as_strided) keeps (9, 15, 4).
    var_node_1 = torch.as_strided(var_node_2, (9, 15, 4), (60, 4, 1))
    
    var_node_7 = arg_1 # size=(20, 15), stride=(15, 1), dtype=int64, device=cuda
    var_node_8 = arg_2 # size=(18, 15), stride=(15, 1), dtype=int64, device=cuda
    _input_size_var_node_6 = var_node_7.size(0)
    
    # Replacement: torch.randint -> torch.isreal
    # Original: _index_var_node_6 = torch.randint(0, _input_size_var_node_6, (18, 15), device=var_node_7.device)
    # Replacement: _index_var_node_6 = torch.isreal(torch.empty((18, 15), device=var_node_7.device))
    _index_var_node_6 = torch.isreal(torch.empty((18, 15), device=var_node_7.device))
    
    # Replacement: torch.gather -> torch.floor
    # Original: var_node_6 = torch.gather(var_node_7, 0, _index_var_node_6) # size=(18, 15)
    # Replacement: var_node_6 = torch.floor(var_node_7) # size=(20, 15)
    var_node_6 = torch.floor(var_node_7)
    
    # Replacement: torch.chunk -> torch.triu
    # Original: var_node_5 = torch.chunk(var_node_6, 2, dim=0)[0] # size=(9, 15)
    # Replacement: var_node_5 = torch.triu(var_node_6) # size=(20, 15)
    var_node_5 = torch.triu(var_node_6)
    
    # Replacement: torch.mul -> torch.tanh
    # Original: var_node_0 = torch.mul(var_node_1, var_node_5) # size=(9, 15)
    # Replacement: var_node_0 = torch.tanh(var_node_1) # size=(9, 15, 4)
    var_node_0 = torch.tanh(var_node_1)
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Replacement: torch.tensor -> tf.linalg.diag
# Original: sentinel = torch.tensor(1.0, requires_grad=True)
# Note: tf.linalg.diag is a TensorFlow API. To keep the test case runnable in PyTorch,
# we use the PyTorch equivalent torch.diag.
# Replacement: sentinel = torch.diag(torch.tensor([1.0], requires_grad=True))
sentinel = torch.diag(torch.tensor([1.0], requires_grad=True))

# Determine device (CUDA or CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Replacement: torch.as_strided -> torch.empty
# Replacement: torch.randint -> torch.isreal
# Original: arg_0 = torch.as_strided(torch.randint(5, 30, (484,)).to(torch.int32), (9, 1, 15, 4), (60, 60, 0, 1))
# Replacement: arg_0 = torch.empty((9, 1, 15, 4), dtype=torch.int32, device=device)
arg_0 = torch.empty((9, 1, 15, 4), dtype=torch.int32, device=device)

# Original: arg_1 = torch.as_strided(torch.randint(5, 30, (300,)).to(torch.int64), (20, 15), (15, 1))
# Replacement: arg_1 = torch.empty((20, 15), dtype=torch.int64, device=device)
arg_1 = torch.empty((20, 15), dtype=torch.int64, device=device)

# Original: arg_2 = torch.as_strided(torch.randint(5, 30, (270,)).to(torch.int64), (18, 15), (15, 1))
# Replacement: arg_2 = torch.empty((18, 15), dtype=torch.int64, device=device)
arg_2 = torch.empty((18, 15), dtype=torch.int64, device=device)

args = (arg_0, arg_1, arg_2) + (sentinel,)

# Run eager mode
result_original = fuzzed_program(*args)
print(' eager success')

# Run compiled mode
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' compile success')