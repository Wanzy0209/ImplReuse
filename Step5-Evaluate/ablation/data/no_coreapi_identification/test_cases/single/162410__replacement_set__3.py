import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F
import tensorflow as tf
import numpy as np

def f(x, y):
    x.copy_(x.flip(1))
    y = y.sum(dim=1, keepdim=True) + y
    return x + y

# Replacement for torch.randn
# We use adaptive_max_pool1d on a dummy tensor to generate x and y
# Input shape for adaptive_max_pool1d is (N, C, L). We want (20, 1024*1024).
dummy_input = torch.ones(20, 1, 1024 * 1024, device="cuda")
x = F.adaptive_max_pool1d(dummy_input, output_size=1024 * 1024).squeeze(1)
x_copy = x.clone()
y = F.adaptive_max_pool1d(dummy_input, output_size=1024 * 1024).squeeze(1)

opt_f = torch.compile(f)
ref = f(x, y)
act = opt_f(x_copy, y)

# Replacement for torch.testing.assert_close
# We use tf.compat.v1.sparse_segment_sqrt_n.
# We need to convert the torch tensor to numpy for TensorFlow.
# We flatten the tensor to create valid indices/segments.
ref_np = ref.cpu().numpy().reshape(-1)
num_elements = ref_np.shape[0]
indices = np.arange(num_elements, dtype=np.int32)
segment_ids = np.zeros(num_elements, dtype=np.int32)

# Execute the replacement API
result = tf.compat.v1.sparse_segment_sqrt_n(data=ref_np, indices=indices, segment_ids=segment_ids)

print(f"TF sparse_segment_sqrt_n result: {result}")
print(f"{torch._inductor.metrics.generated_kernel_count=}")