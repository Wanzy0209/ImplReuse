```python
import tensorflow as tf

GB = 1024*1024*1024
# Conversion: In TensorFlow, devices are specified by strings (e.g., '/GPU:0')
device = '/GPU:0'


def dump_mem():
    # Conversion: tf.config.experimental.get_memory_info returns 'current' and 'peak' usage in bytes.
    # PyTorch's mem_get_info returns (free, total). We adapt the output to show TF's available metrics.
    try:
        mem_info = tf.config.experimental.get_memory_info(device)
        allocated = mem_info['current']
        peak = mem_info['peak']
        # Note: 'Total' memory is not directly exposed via this API, so we print allocated/peak instead.
        print(f"allocated: {allocated / (GB):.2f} GiB, peak: {peak / (GB):.2f} GiB")
    except RuntimeError as e:
        # Handle cases where GPU might not be available or memory info cannot be retrieved
        print(f"Could not retrieve memory info: {e}")


def alloc_1gb():
    # Conversion: torch.empty allocates uninitialized memory. 
    # TensorFlow initializes tensors (e.g., with zeros). We use tf.Variable to hold the state.
    with tf.device(device):
        return tf.Variable(tf.zeros([GB], dtype=tf.int8))


print("before allocation")
dump_mem()

t0 = alloc_1gb()
print("t0 allocated")
dump_mem()

# Conversion: TensorFlow manages memory pools internally (BFC Allocator) and does not expose 
# explicit user-defined memory pools (MemPool) or context managers to switch them like PyTorch.
# We perform the allocation directly here.
# mpool = torch.cuda.MemPool()
# with torch.cuda.use_mem_pool(mpool):
t1 = alloc_1gb()
print("t1 allocated")
dump_mem()

t0 = None
t1 = None
print("t0 and t1 freed")
dump_mem()

# Conversion: TensorFlow manages the memory cache automatically. 
# There is no direct equivalent to torch.cuda.empty_cache() to force immediate release to the OS.
# The allocator will reuse this memory for future TensorFlow operations.
# torch.cuda.empty_cache()
print("cache emptied (automatic in TF)")
dump_mem()
```