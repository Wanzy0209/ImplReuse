```python
# Conversion: TensorFlow does not have direct public equivalents for PyTorch's internal sharding metadata classes.
# We define equivalent structures to replicate the logic for the purpose of this test case.

import tensorflow as tf
import time
from dataclasses import dataclass
from typing import List

# Conversion: Define ShardMetadata similar to torch.distributed._shard.metadata.ShardMetadata
@dataclass
class ShardMetadata:
    shard_offsets: List[int]
    shard_sizes: List[int]
    placement: str

# Conversion: Implement validation logic similar to torch.distributed._shard.sharding_spec._internals.validate_non_overlapping_shards_metadata
def validate_non_overlapping_shards_metadata(shards):
    """
    Validates that the provided shards do not overlap.
    This mimics the behavior of the PyTorch internal function.
    """
    n = len(shards)
    for i in range(n):
        for j in range(i + 1, n):
            s1 = shards[i]
            s2 = shards[j]
            
            # Check overlap in all dimensions
            overlap = True
            for dim in range(len(s1.shard_offsets)):
                start1, size1 = s1.shard_offsets[dim], s1.shard_sizes[dim]
                start2, size2 = s2.shard_offsets[dim], s2.shard_sizes[dim]
                end1 = start1 + size1
                end2 = start2 + size2
                
                # If intervals [start, end) do not intersect, shards are disjoint in this dimension
                if end1 <= start2 or end2 <= start1:
                    overlap = False
                    break
            
            if overlap:
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
        
        # Conversion: Using the defined ShardMetadata class
        shard = ShardMetadata(
            shard_offsets=[offset_0, offset_1],
            shard_sizes=[shard_height, shard_width],
            placement=f"rank:{i}/cuda:{i}" # Keeping original placement format for consistency
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