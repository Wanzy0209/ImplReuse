```python
import tensorflow as tf

def triton_tem_fused_rms_norm_pipeline(input_tensor, weight, multiplier, split_sizes, perm, shape, epsilon=1e-5):
    """
    TensorFlow implementation corresponding to the Triton fused kernel:
    triton_tem_fused__fused_rms_norm__to_copy_mul_split_with_sizes_transpose_view_zeros_26
    """
    # 1. fused_rms_norm
    # Calculate Root Mean Square along the last dimension
    variance = tf.reduce_mean(tf.square(input_tensor), axis=-1, keepdims=True)
    rms = tf.sqrt(variance + epsilon)
    
    # Normalize and scale by weight
    x = input_tensor / rms * weight
    
    # 2. to_copy
    # Cast to float32 (common target for 'to' in mixed precision contexts)
    x = tf.cast(x, tf.float32)
    
    # 3. mul
    # Element-wise multiplication
    x = tf.multiply(x, multiplier)
    
    # 4. split_with_sizes
    # Split the tensor along the last dimension into specific sizes
    split_tensors = tf.split(x, num_or_size_splits=split_sizes, axis=-1)
    # Assuming the pipeline continues with the first split
    x = split_tensors[0]
    
    # 5. transpose
    # Permute dimensions according to the permutation vector
    x = tf.transpose(x, perm=perm)
    
    # 6. view
    # Reshape the tensor to the target shape
    x = tf.reshape(x, shape)
    
    # 7. zeros
    # Create a tensor of zeros with the same shape and dtype as x
    zeros_tensor = tf.zeros_like(x)
    
    return x, zeros_tensor
```