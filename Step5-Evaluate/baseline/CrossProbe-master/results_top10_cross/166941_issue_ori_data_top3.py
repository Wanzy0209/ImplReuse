```python
import tensorflow as tf
import time
from typing import List, NamedTuple

# Conversion: TensorFlow does not have a direct equivalent to PyTorch's ShardMetadata class.
# We define a NamedTuple to replicate the structure for this translation.
class ShardMetadata(NamedTuple):
    shard_offsets: List[int]
    shard_sizes: List[int]
    placement: str

# Conversion: TensorFlow does not have a direct equivalent to 
# validate_non_overlapping_shards_metadata. We implement the logic manually
# to check if any shards overlap in the 2D space.
def validate_non_overlapping_shards_metadata(shards: List[ShardMetadata]):
    num_shards = len(shards)
    # Extract offsets and sizes into lists for easier access
    offsets = [s.shard_offsets for s in shards]
    sizes = [s.shard_sizes for s in shards]
    
    for i in range(num_shards):
        for j in range(i + 1, num_shards):
            # Check for overlap in dimension 0
            # Overlap exists if: start_a < end_b AND start_b < end_a
            end_0_i = offsets[i][0] + sizes[i][0]
            end_0_j = offsets[j][0] + sizes[j][0]
            overlap_0 = offsets[i][0] < end_0_j and offsets[j][0] < end_0_i
            
            # Check for overlap in dimension 1
            end_1_i = offsets[i][1] + sizes[i][1]
            end_1_j = offsets[j][1] + sizes[j][1]
            overlap_1 = offsets[i][1] < end_1_j and offsets[j][1] < end_1_i
            
            if overlap_0 and overlap_1:
                raise ValueError(f"Shards {i} and {j} overlap.")


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