```python
import tensorflow as tf

GB = 1024*1024*1024
# Conversion: PyTorch uses torch.device objects, TF uses device strings like '/GPU:0'
device = '/GPU:0'


def dump_mem():
    # Conversion: torch.cuda.mem_get_info() returns (free, total).
    # TF's tf.config.experimental.get_memory_info returns {'current': bytes, 'peak': bytes}.
    # 'Total' memory is not directly exposed via the public TF API without external libraries.
    # We print current allocation here as the closest equivalent.
    mem_info = tf.config.experimental.get_memory_info(device)
    print(f"current: {mem_info['current'] / (GB):.2f} GiB, peak: {mem_info['peak'] / (GB):.2f} GiB")


def alloc_1gb():
    # Conversion: torch.empty creates an uninitialized tensor.
    # tf.empty is the direct equivalent. We wrap in tf.Variable to ensure
    # the memory is allocated and held in eager execution.
    return tf.Variable(tf.empty((GB,), dtype=tf.int8))


print("before allocation")
dump_mem()

t0 = alloc_1gb()
print("t0 allocated")
dump_mem()

# Conversion: torch.cuda.MemPool and torch.cuda.use_mem_pool are not available in TensorFlow.
# TensorFlow uses the BFC (Buddy Fragmentation Check) allocator and does not support
# custom memory pools in this manner. We proceed with allocation directly.
# mpool = torch.cuda.MemPool()
# with torch.cuda.use_mem_pool(mpool):
t1 = alloc_1gb()
print("t1 allocated")
dump_mem()

t0 = None
t1 = None
print("t0 and t1 freed")
dump_mem()

# Conversion: torch.cuda.empty_cache() forces the cache to clear.
# TensorFlow manages memory internally and does not provide a public API
# to force the allocator to return memory to the OS immediately.
# torch.cuda.empty_cache()
print("cache emptied")
dump_mem()
```