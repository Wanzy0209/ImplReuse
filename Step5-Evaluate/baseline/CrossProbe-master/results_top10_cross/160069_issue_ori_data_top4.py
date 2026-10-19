```python
import tensorflow as tf

GB = 1024*1024*1024
# Conversion: torch.device('cuda') -> TensorFlow device string
device_name = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'


def dump_mem():
    # Conversion: torch.cuda.mem_get_info() -> tf.config.experimental.get_memory_info()
    # Note: TF returns current and peak usage bytes, not free/total directly.
    if device_name == '/GPU:0':
        mem_info = tf.config.experimental.get_memory_info('GPU:0')
        current = mem_info['current']
        peak = mem_info['peak']
        # Printing current usage instead of free, as TF manages memory pre-allocation differently
        print(f"current: {current / (GB):.2f} GiB, peak: {peak / (GB):.2f} GiB")
    else:
        print("Memory info not available on CPU")


def alloc_1gb():
    # Conversion: torch.empty -> tf.empty
    # Conversion: device argument -> tf.device context manager
    with tf.device(device_name):
        return tf.empty([GB], dtype=tf.int8)


print("before allocation")
dump_mem()

t0 = alloc_1gb()
print("t0 allocated")
dump_mem()

# Conversion: torch.cuda.MemPool and torch.cuda.use_mem_pool
# TensorFlow does not expose explicit user-level memory pools in the public API.
# Memory is managed automatically by the BFC allocator.
# We perform the allocation directly without the pool context.
t1 = alloc_1gb()
print("t1 allocated")
dump_mem()

t0 = None
t1 = None
print("t0 and t1 freed")
dump_mem()

# Conversion: torch.cuda.empty_cache()
# TensorFlow manages memory retention internally and does not provide a direct
# public API to force the allocator to release all cached memory to the OS.
print("cache emptied (TF manages memory retention automatically)")
dump_mem()
```