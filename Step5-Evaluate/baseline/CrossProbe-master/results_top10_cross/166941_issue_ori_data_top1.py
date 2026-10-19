```python
import time
from dataclasses import dataclass
from typing import List

# Conversion: Define ShardMetadata as a dataclass. 
# TensorFlow's DTensor uses Layout and Mesh objects for sharding, 
# but we replicate the PyTorch structure here to maintain the logic of the script.
@dataclass
class ShardMetadata:
    shard_offsets: List[int]
    shard_sizes: List[int]
    placement: str

# Conversion: Implement the validation logic. 
# TensorFlow handles non-overlapping checks internally via Layout definitions.
# This function replicates the explicit validation logic from the PyTorch source.
def validate_non_overlapping_shards_metadata(shards: List[ShardMetadata]):
    num_shards = len(shards)
    for i in range(num_shards):
        for j in range(i + 1, num_shards):
            s1 = shards[i]
            s2 = shards[j]
            
            # Check for overlap in all dimensions
            overlap = True
            for dim in range(len(s1.shard_offsets)):
                o1, sz1 = s1.shard_offsets[dim], s1.shard_sizes[dim]
                o2, sz2 = s2.shard_offsets[dim], s2.shard_sizes[dim]
                
                # If intervals [o1, o1+sz1) and [o2, o2+sz2) do not intersect, no overlap
                if (o1 + sz1 <= o2) or (o2 + sz2 <= o1):
                    overlap = False
                    break
            
            if overlap:
                raise ValueError(f"Overlapping shards found: {s1} and {s2}")


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