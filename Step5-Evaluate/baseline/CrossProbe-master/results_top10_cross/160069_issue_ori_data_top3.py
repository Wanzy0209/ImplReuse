```python
import tensorflow as tf
from contextlib import nullcontext

GB = 1024*1024*1024
# Conversion: torch.device('cuda') -> TF device string
device = '/GPU:0'


def dump_mem():
    # Conversion: torch.cuda.mem_get_info returns (free, total).
    # TF get_memory_info returns {'current': used, 'peak': used}.
    # We adapt the output to show current usage as 'free' and 'total' are not directly available.
    mem_info = tf.config.experimental.get_memory_info(device)
    current = mem_info['current']
    print(f"current usage: {current / (GB):.2f} GiB")


def alloc_1gb():
    # Conversion: torch.empty -> tf.empty
    with tf.device(device):
        return tf.empty((GB,), dtype=tf.int8)


print("before allocation")
dump_mem()

t0 = alloc_1gb()
print("t0 allocated")
dump_mem()

# Conversion: torch.cuda.MemPool/use_mem_pool have no direct equivalent in TF.
# TF manages memory pools internally. Using nullcontext to preserve structure.
# mpool = torch.cuda.MemPool()
with nullcontext(): # Replaces torch.cuda.use_mem_pool(mpool)
    t1 = alloc_1gb()
    print("t1 allocated")
    dump_mem()

t0 = None
t1 = None
print("t0 and t1 freed")
dump_mem()

# Conversion: torch.cuda.empty_cache() has no direct equivalent.
# TF manages memory automatically.
# torch.cuda.empty_cache()
print("cache emptied (TF manages automatically)")
dump_mem()
```