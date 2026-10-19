import torch
import tensorflow as tf
import numpy as np

# Replacement: torch.manual_seed -> torch.argmax
# Original: torch.manual_seed(1337)
_ = torch.argmax(torch.tensor([1337]))

# Replacement: torch.compile -> tf.keras.losses.hinge
# Replacement: nn.functional -> tf.keras.backend.resize_volumes
# Original:
# @torch.compile()
# def vec_norm(e_dist):
#     return torch.nn.functional.normalize(e_dist)
def vec_norm(e_dist):
    # Convert torch tensor to tf tensor for TF ops
    e_dist_tf = tf.convert_to_tensor(e_dist.cpu().numpy())
    # Reshape for resize_volumes (5D required)
    e_dist_5d = tf.expand_dims(tf.expand_dims(tf.expand_dims(e_dist_tf, axis=1), axis=1), axis=1)
    
    # Replacement: nn.functional -> tf.keras.backend.resize_volumes
    resized = tf.keras.backend.resize_volumes(e_dist_5d, 1, 1, 1, data_format='channels_last')
    resized = tf.squeeze(resized, axis=[1, 2, 3])
    
    # Replacement: torch.compile -> tf.keras.losses.hinge
    # We use reduction='none' to keep dimensions similar to normalize
    y_true = tf.ones_like(resized)
    result = tf.keras.losses.hinge(y_true, resized, reduction='none')
    
    # Convert back to torch
    return torch.from_numpy(result.numpy()).to(e_dist.device)

# Original:
# def vec_norm_without_compile(e_dist):
#     return torch.nn.functional.normalize(e_dist)
def vec_norm_without_compile(e_dist):
    e_dist_tf = tf.convert_to_tensor(e_dist.cpu().numpy())
    e_dist_5d = tf.expand_dims(tf.expand_dims(tf.expand_dims(e_dist_tf, axis=1), axis=1), axis=1)
    # Replacement: nn.functional -> tf.keras.backend.resize_volumes
    resized = tf.keras.backend.resize_volumes(e_dist_5d, 1, 1, 1, data_format='channels_last')
    resized = tf.squeeze(resized, axis=[1, 2, 3])
    return torch.from_numpy(resized.numpy()).to(e_dist.device)

# Original:
# device='cuda'
# c=torch.tensor([[3.799999 ,0.0, 0.0]],device=device,dtype=torch.float32)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create raw data
raw_data = [[3.799999, 0.0, 0.0]]

# Replacement: torch.float32 -> tf.keras.backend.resize_volumes
# We use resize_volumes to process the raw data (mimicking the type casting/initialization step)
raw_tf = tf.constant(raw_data)
raw_tf_5d = tf.expand_dims(tf.expand_dims(tf.expand_dims(raw_tf, axis=1), axis=1), axis=1)
processed_tf = tf.keras.backend.resize_volumes(raw_tf_5d, 1, 1, 1, data_format='channels_last')
processed_tf = tf.squeeze(processed_tf, axis=[1, 2, 3])

# Convert to torch
processed_torch = torch.from_numpy(processed_tf.numpy()).to(device)

# Replacement: torch.tensor -> torch.cummax
# We use cummax to finalize the tensor creation
c, _ = torch.cummax(processed_torch, dim=0)

print("Input vector", [x.item() for x in c[0]])

xyz = vec_norm(c)
# Replacement: torch.acos -> torch.reciprocal
print("Normalized vector (compile):", [x.item() for x in xyz[0]], "torch.reciprocal of component 0", torch.reciprocal(xyz[0,0]).item())

xyz = vec_norm_without_compile(c)
print("Normalized vector (without compile):", [x.item() for x in xyz[0]], "torch.reciprocal of component 0", torch.reciprocal(xyz[0,0]).item())

# Check for the bug condition (norm > 1) to preserve the original test logic
norm_compile = torch.norm(xyz)
print("Norm (compile):", norm_compile.item())
if norm_compile > 1.0:
    print("Bug detected: Norm > 1")