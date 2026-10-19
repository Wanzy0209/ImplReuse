```python
import time
import dataclasses

# Conversion: TensorFlow does not have a direct equivalent to PyTorch's ShardMetadata class
# or the internal validation function. We define a dataclass to mimic the structure
# and implement the validation logic to preserve the benchmark's behavior.
@dataclasses.dataclass
class ShardMetadata:
    shard_offsets: list
    shard_sizes: list
    placement: str

def validate_non_overlapping_shards_metadata(shards):
    """
    Validates that the provided list of ShardMetadata objects do not overlap.
    This replicates the logic of the PyTorch internal function.
    """
    # Simple O(N^2) check for overlaps in 2D
    n = len(shards)
    for i in range(n):
        for j in range(i + 1, n):
            s1 = shards[i]
            s2 = shards[j]

            # Check dimension 0
            # Intervals: [s1.offsets[0], s1.offsets[0] + s1.sizes[0])
            #            [s2.offsets[0], s2.offsets[0] + s2.sizes[0])
            overlap_0 = not (s1.shard_offsets[0] + s1.shard_sizes[0] <= s2.shard_offsets[0] or
                             s2.shard_offsets[0] + s2.shard_sizes[0] <= s1.shard_offsets[0])

            # Check dimension 1
            overlap_1 = not (s1.shard_offsets[1] + s1.shard_sizes[1] <= s2.shard_offsets[1] or
                             s2.shard_offsets[1] + s2.shard_sizes[1] <= s1.shard_offsets[1])

            if overlap_0 and overlap_1:
                raise ValueError(f"Overlapping shards detected: {s1} and {s2}")

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
        
        # Conversion: Using the defined ShardMetadata dataclass
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