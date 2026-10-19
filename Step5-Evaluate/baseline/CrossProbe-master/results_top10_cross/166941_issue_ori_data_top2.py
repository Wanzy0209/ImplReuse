```python
import tensorflow as tf
import time
from dataclasses import dataclass

# Conversion: TensorFlow does not have a direct public equivalent for ShardMetadata.
# We define a dataclass to mimic the structure used in the PyTorch benchmark.
@dataclass
class ShardMetadata:
    shard_offsets: list
    shard_sizes: list
    placement: str

# Conversion: TensorFlow does not expose a specific internal API for validating shard metadata overlaps.
# We implement the logic to check for geometric overlaps between shards.
def validate_non_overlapping_shards_metadata(shards):
    for i in range(len(shards)):
        for j in range(i + 1, len(shards)):
            s1 = shards[i]
            s2 = shards[j]
            
            # Check overlap in dimension 0
            s1_end_0 = s1.shard_offsets[0] + s1.shard_sizes[0]
            s2_end_0 = s2.shard_offsets[0] + s2.shard_sizes[0]
            overlap_0 = not (s1_end_0 <= s2.shard_offsets[0] or s2_end_0 <= s1.shard_offsets[0])
            
            # Check overlap in dimension 1
            s1_end_1 = s1.shard_offsets[1] + s1.shard_sizes[1]
            s2_end_1 = s2.shard_offsets[1] + s2.shard_sizes[1]
            overlap_1 = not (s1_end_1 <= s2.shard_offsets[1] or s2_end_1 <= s1.shard_offsets[1])
            
            if overlap_0 and overlap_1:
                raise ValueError(f"Overlapping shards detected between index {i} and {j}")

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
        
        # Conversion: Adapted placement string to TensorFlow device naming convention
        placement = f"/job:worker/task:{i}/device:GPU:{i % 8}"
        
        shard = ShardMetadata(
            shard_offsets=[offset_0, offset_1],
            shard_sizes=[shard_height, shard_width],
            placement=placement
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