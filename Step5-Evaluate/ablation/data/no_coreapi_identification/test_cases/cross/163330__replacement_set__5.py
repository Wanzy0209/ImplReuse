import torch
import torch.nn.functional as F
import torch.nn as nn
import tensorflow as tf

# Replacement for torch.distributed.init_process_group("nccl")
# torch.nn.functional.kl_div
# We use dummy tensors to satisfy the function signature.
input_tensor = torch.randn(1, 10, requires_grad=True)
target_tensor = torch.randn(1, 10)
F.kl_div(input_tensor, target_tensor)

# Replacement for init_device_mesh
# tf.nn.ctc_beam_search_decoder
# We map the mesh shape (1, 4, 2) to logits shape (batch=1, time=4, classes=2)
# to preserve the dimensional logic of the original test.
logits1 = tf.random.uniform((1, 4, 2), dtype=tf.float32)
seq_len1 = tf.constant([4], dtype=tf.int32)
mesh1 = tf.nn.ctc_beam_search_decoder(logits1, seq_len1)

# mesh1_c = mesh1["c"]
# The original code accesses a sub-mesh via string key.
# The replacement API returns a tuple (decoded, log_prob).
# To make the code runnable and preserve the logic of accessing a component,
# we access the first element (decoded sparse tensor).
mesh1_c = mesh1[0]

# Replacement for torch.distributed.get_rank() == 0
# torch.nn.utils.parameters_to_vector
# We create a dummy parameter and vectorize it to mimic the rank check condition.
param = nn.Parameter(torch.randn(10))
rank_check = torch.nn.utils.parameters_to_vector([param])

# Original logic: if torch.distributed.get_rank() == 0: print(...)
# We check if the vector is non-empty (always true) to mimic the block execution.
if rank_check.numel() > 0:
    # Replacement for _mesh_resources.get_root_mesh(mesh1_c)
    # tf.nn.ctc_beam_search_decoder
    # mesh1_c is a SparseTensor. We convert it to dense to use as valid input for ctc.
    dense_c = tf.sparse.to_dense(mesh1_c)
    root1 = tf.nn.ctc_beam_search_decoder(dense_c, seq_len1)
    # Print shape/values to mimic original output
    print(root1[0].values)

# Replacement for init_device_mesh (mesh2)
# tf.nn.ctc_beam_search_decoder
# Shape (2, 2, 2) -> (batch=2, time=2, classes=2)
logits2 = tf.random.uniform((2, 2, 2), dtype=tf.float32)
seq_len2 = tf.constant([2, 2], dtype=tf.int32)
mesh2 = tf.nn.ctc_beam_search_decoder(logits2, seq_len2)

mesh2_c = mesh2[0]

if rank_check.numel() > 0:
    # Check root of mesh1_c again.
    # Original bug: returns mesh2 instead of mesh1.
    # Here we call ctc on mesh1_c (dense) again.
    root1_check = tf.nn.ctc_beam_search_decoder(dense_c, seq_len1)
    print(root1_check[0].values)

# Original assertion: assert _mesh_resources.get_root_mesh(mesh1_c) is mesh1
# We assert the result of the second call is the same object as the first.
# Note: This assertion will likely fail because ctc_beam_search_decoder returns new objects,
# but it preserves the original test case logic of checking object identity.
assert root1_check is mesh1

# Replacement for torch.distributed.destroy_process_group
# tf.nn.ctc_beam_search_decoder
tf.nn.ctc_beam_search_decoder(logits1, seq_len1)