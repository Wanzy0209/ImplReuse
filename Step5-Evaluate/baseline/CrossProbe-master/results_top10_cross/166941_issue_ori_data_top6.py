```python
import time
from dataclasses import dataclass
from typing import List

# Conversion: PyTorch ShardMetadata is a dataclass. We define a similar structure here.
@dataclass
class ShardMetadata:
    shard_offsets: List[int]
    shard_sizes: List[int]
    placement: str

# Conversion: TensorFlow does not have a direct equivalent for this specific validation utility.
# We implement the logic to check for overlapping shards in 2D.
def validate_non_overlapping_shards_metadata(shards: List[ShardMetadata]):
    num_shards = len(shards)
    for i in range(num_shards):
        s1 = shards[i]
        r1_start, c1_start = s1.shard_offsets
        r1_end = r1_start + s1.shard_sizes[0]
        c1_end = c1_start + s1.shard_sizes[1]

        for j in range(i + 1, num_shards):
            s2 = shards[j]
            r2_start, c2_start = s2.shard_offsets
            r2_end = r2_start + s2.shard_sizes[0]
            c2_end = c2_start + s2.shard_sizes[1]

            # Check intersection
            if (r1_start < r2_end and r2_start < r1_end and
                c1_start < c2_end and c2_start < c1_end):
                raise ValueError(f"Shards {i} and {j} overlap")

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