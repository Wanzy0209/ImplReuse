```python
import tensorflow as tf

# Conversion Note: TensorFlow tensors are immutable and do not support explicit custom strides 
# or uninitialized memory allocation in the same way as PyTorch. 
# We simulate this behavior using a 1D buffer and a wrapper class to track shape and stride metadata.

class StridedTensor:
    def __init__(self, data, shape, stride):
        self.data = data
        self._shape = tuple(shape)
        self._stride = tuple(stride)

    def stride(self):
        return self._stride

def empty_strided(size, stride):
    # Calculate the total number of elements required in the 1D buffer
    # to accommodate the non-contiguous stride.
    # Formula: sum((size[i] - 1) * stride[i]) + 1
    offset = sum((s - 1) * st for s, st in zip(size, stride))
    total_elements = offset + 1
    
    # PyTorch's empty is uninitialized. TF doesn't expose uninitialized tensors in Python,
    # so we use zeros as a placeholder for the allocation.
    data = tf.zeros(total_elements, dtype=tf.float32)
    return StridedTensor(data, size, stride)

def rand_like(input_tensor):
    # Create a new random buffer with the same underlying size
    new_data = tf.random.uniform(shape=input_tensor.data.shape, minval=0, maxval=1)
    # Preserve the shape and stride metadata from the input
    return StridedTensor(new_data, input_tensor._shape, input_tensor._stride)

input_tensor  = empty_strided(size = [1792, 1899, 160], stride = (303872, 160, 1))
output_tensor = rand_like(input_tensor)
assert input_tensor.stride() == output_tensor.stride()
```