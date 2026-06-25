```python
import time
import tensorflow as tf

# Conversion Note: TensorFlow (specifically tf.dtensor) does not expose a direct public API 
# equivalent to PyTorch's internal 'ShardMetadata' class or the specific 
# 'validate_non_overlapping_shards_metadata' utility. 
# Below, we define a compatible class and validation logic to replicate the behavior 
# for the purpose of this benchmark.

class ShardMetadata:
    """
    Mimics torch.distributed._shard.metadata.ShardMetadata
    """
    def __init__(self, shard_offsets, shard_sizes, placement):
        self.shard_offsets = shard_offsets
        self.shard_sizes = shard_sizes
        self.placement = placement

def validate_non_overlapping_shards_metadata(shards):
    """
    Mimics torch.distributed._shard.sharding_spec._internals.validate_non_overlapping_shards_metadata
    Checks if any shards in the list overlap spatially.
    """
    num_shards = len(shards)
    for i in range(num_shards):
        for j in range(i + 1, num_shards):
            s1 = shards[i]
            s2 = shards[j]
            
            # Check overlap in all dimensions
            is_overlapping = True
            for dim in range(len(s1.shard_offsets)):
                s1_start = s1.shard_offsets[dim]
                s1_end = s1_start + s1.shard_sizes[dim]
                s2_start = s2.shard_offsets[dim]
                s2_end = s2_start + s2.shard_sizes[dim]
                
                # If intervals are disjoint in any dimension, shards do not overlap
                if s1_end <= s2_start or s2_end <= s1_start:
                    is_overlapping = False
                    break
            
            if is_overlapping:
                raise ValueError(f"Shards {i} and {j} overlap: {s1} vs {s2}")

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