```python
import tensorflow as tf

GB = 1024*1024*1024
# TensorFlow uses device strings (e.g., '/GPU:0') instead of torch.device objects
device = '/GPU:0'


def dump_mem():
    # PyTorch: torch.cuda.mem_get_info() returns (free, total)
    # TensorFlow: get_memory_info returns a dict with 'current' (allocated) and 'peak' bytes.
    # 'free' and 'total' are not directly exposed via the high-level API in the same way.
    mem_info = tf.config.experimental.get_memory_info(device)
    print(f"allocated: {mem_info['current'] / (GB):.2f} GiB, peak: {mem_info['peak'] / (GB):.2f} GiB")


def alloc_1gb():
    # PyTorch: torch.empty((GB,), dtype=torch.int8, device=device)
    # TensorFlow: tf.empty creates an uninitialized tensor. Shape is a list.
    with tf.device(device):
        return tf.empty([GB], dtype=tf.int8)


print("before allocation")
dump_mem()

t0 = alloc_1gb()
print("t0 allocated")
dump_mem()

# PyTorch: mpool = torch.cuda.MemPool()
# PyTorch: with torch.cuda.use_mem_pool(mpool):
# TensorFlow does not expose a user-facing Memory Pool API like PyTorch's CUDA graph pools.
# TF manages memory internally via the BFC allocator. We proceed with the allocation directly.
t1 = alloc_1gb()
print("t1 allocated")
dump_mem()

t0 = None
t1 = None
print("t0 and t1 freed")
dump_mem()

# PyTorch: torch.cuda.empty_cache()
# TensorFlow: tf.keras.backend.clear_session() resets the global state and releases memory.
tf.keras.backend.clear_session()
print("cache cleared")
dump_mem()
```