```python
import tensorflow as tf

GB = 1024*1024*1024
# PyTorch uses 'cuda', TensorFlow uses device strings like '/GPU:0'
device_name = '/GPU:0'

# Configure memory growth to mimic PyTorch's dynamic allocation behavior
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def dump_mem():
    # PyTorch: torch.cuda.mem_get_info() returns (free, total)
    # TensorFlow: tf.config.experimental.get_memory_info returns {'current': ..., 'peak': ...}
    # Note: TF does not easily expose 'total' physical memory or 'free' memory directly.
    # We print 'current' (allocated) and 'peak' here as the closest equivalent.
    if gpus:
        mem_info = tf.config.experimental.get_memory_info('GPU:0')
        print(f"current: {mem_info['current'] / (GB):.2f} GiB, peak: {mem_info['peak'] / (GB):.2f} GiB")
    else:
        print("Memory info not available on CPU")

def alloc_1gb():
    # PyTorch: torch.empty((GB,), dtype=torch.int8, device=device)
    # TensorFlow: tf.empty creates uninitialized tensors (available in TF 2.12+).
    # We must use the tf.device context manager as tf.empty doesn't take a device argument.
    with tf.device(device_name):
        return tf.empty([GB], dtype=tf.int8)

print("before allocation")
dump_mem()

# Allocation needs to happen inside the device context
with tf.device(device_name):
    t0 = alloc_1gb()
print("t0 allocated")
dump_mem()

# PyTorch: mpool = torch.cuda.MemPool()
# PyTorch: with torch.cuda.use_mem_pool(mpool):
# TensorFlow does not have a direct equivalent for user-defined memory pools.
# Memory management is handled automatically by the runtime.
# We simply perform the allocation to maintain the script flow.
print("Allocating t1 (no direct MemPool equivalent in TF)")
with tf.device(device_name):
    t1 = alloc_1gb()
print("t1 allocated")
dump_mem()

t0 = None
t1 = None
print("t0 and t1 freed")
dump_mem()

# PyTorch: torch.cuda.empty_cache()
# TensorFlow manages memory automatically and does not expose an empty_cache() function.
# Memory is typically returned to the OS when the Python object is deleted and growth is enabled.
print("cache emptied (automatic in TF)")
dump_mem()
```