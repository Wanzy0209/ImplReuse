```python
import tensorflow as tf

# Conversion: torch.rand(2, device=torch.device("cuda:0"))
# In TensorFlow, device placement is handled via context managers.
# We assume the environment has at least 2 GPUs available.
with tf.device('/device:GPU:0'):
    tensor1 = tf.random.uniform((2,), dtype=tf.float32)

# Conversion: torch.rand(2, device=torch.device("cuda:1"))
with tf.device('/device:GPU:1'):
    tensor2 = tf.random.uniform((2,), dtype=tf.float32)

print(f"tensor1: {tensor1}, tensor2: {tensor2}")

# Conversion: torch.nn.parallel.comm.gather
# TensorFlow does not have a direct equivalent for the specific PyTorch gather function
# outside of distributed strategies. We implement the logic by moving tensors
# to the destination device and concatenating them.
def gather_tensors(tensor_list, destination_device):
    gathered = []
    with tf.device(destination_device):
        for t in tensor_list:
            # tf.identity ensures the tensor is copied to the destination device
            gathered.append(tf.identity(t))
        return tf.concat(gathered, axis=0)

# Conversion: destination = torch.device("cuda:0")
print(f'combined tensor - Cuda: {gather_tensors([tensor1, tensor2], "/device:GPU:0")}')
# Conversion: destination = torch.device("cpu")
print(f'combined tensor - CPU: {gather_tensors([tensor1, tensor2], "/device:CPU:0")}')
```