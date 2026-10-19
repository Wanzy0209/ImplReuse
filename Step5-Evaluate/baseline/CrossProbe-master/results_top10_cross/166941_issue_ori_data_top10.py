```python
import tensorflow as tf
import time

# Conversion: TensorFlow does not have a direct public equivalent for ShardMetadata.
# We define a simple class to mimic the structure for the benchmark.
class ShardMetadata:
    def __init__(self, shard_offsets, shard_sizes, placement):
        self.shard_offsets = shard_offsets
        self.shard_sizes = shard_sizes
        self.placement = placement

# Conversion: TensorFlow handles sharding internally via tf.distribute.Strategy
# and does not expose a public API to manually validate shard metadata overlaps.
# We implement a basic validation logic here to replicate the functionality.
def validate_non_overlapping_shards_metadata(shards):
    n = len(shards)
    for i in range(n):
        for j in range(i + 1, n):
            s1 = shards[i]
            s2 = shards[j]
            
            # Check for overlap in dimension 0
            overlap_0 = not (s1.shard_offsets[0] + s1.shard_sizes[0] <= s2.shard_offsets[0] or
                             s2.shard_offsets[0] + s2.shard_sizes[0] <= s1.shard_offsets[0])
            
            # Check for overlap in dimension 1
            overlap_1 = not (s1.shard_offsets[1] + s1.shard_sizes[1] <= s2.shard_offsets[1] or
                             s2.shard_offsets[1] + s2.shard_sizes[1] <= s1.shard_offsets[1])
            
            if overlap_0 and overlap_1:
                raise ValueError(f"Overlapping shards detected: {i} and {j}")


def create_shards_for_tensor_2d(grid_size: int, tensor_shape=(10000, 10000)):
    shards = []
    shard_height = tensor_shape[0] // grid_size
    shard_width = tensor_shape[1] // grid_size
    
    num_shards = grid_size * grid_size
    
    for i in range(num_shards):
        row = i // grid_size
        col = i % grid_size
        
        offset_0 = row * shard_height
        offset_1 = col * shard_width
        
        # Conversion: Preserving the placement string format from the source 
        # to maintain similar string allocation overhead, though TF uses 
        # device strings like '/job:worker/task:0/device:GPU:0'.
        shard = ShardMetadata(
            shard_offsets=[offset_0, offset_1],
            shard_sizes=[shard_height, shard_width],
            placement=f"rank:{i}/cuda:{i}"
        )
        shards.append(shard)
    
    return shards

grid_size = 32
num_tensors = 1024

shards_template = create_shards_for_tensor_2d(grid_size)
print(f"Validating {num_tensors} tensors with {len(shards_template)} shards each...")
print()

start = time.time()

for tensor_idx in range(num_tensors):
    shards = create_shards_for_tensor_2d(grid_size)
    validate_non_overlapping_shards_metadata(shards)
    if (tensor_idx + 1) % 100 == 0:
        elapsed = time.time() - start
        avg_time = elapsed / (tensor_idx + 1)
        estimated_total = avg_time * num_tensors
        print(f"Progress: {tensor_idx+1}/{num_tensors} tensors, "
              f"avg={avg_time*1000:.1f}ms/tensor, "
              f"estimated total={estimated_total:.1f}s")

end = time.time()
total_time = end - start

print()
print("="*80)
print(f"Validation cost {total_time:.1f}s for {num_tensors} tensors")  
print(f"Average: {total_time/num_tensors*1000:.1f} ms per tensor")
print("="*80)

print(f"Total overhead: {total_time:.1f}s = {total_time/60:.1f} minutes")
```