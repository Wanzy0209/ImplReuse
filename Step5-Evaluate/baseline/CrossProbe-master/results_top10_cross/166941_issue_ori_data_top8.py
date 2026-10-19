```python
import tensorflow as tf
import time
import dataclasses

# Conversion: Define ShardMetadata as a dataclass since TensorFlow does not have 
# a direct equivalent to PyTorch's internal ShardMetadata class.
@dataclasses.dataclass
class ShardMetadata:
    shard_offsets: list
    shard_sizes: list
    placement: str

# Conversion: Implement validation logic using TensorFlow operations.
# PyTorch's validate_non_overlapping_shards_metadata is an internal C++ function.
# We replicate the logic here using vectorized TF ops for efficiency.
def validate_non_overlapping_shards_metadata(shards):
    # Extract offsets and sizes from the list of objects
    # Assuming 2D shards based on the creation logic
    offsets = tf.constant([s.shard_offsets for s in shards], dtype=tf.int64)
    sizes = tf.constant([s.shard_sizes for s in shards], dtype=tf.int64)
    
    # Calculate end coordinates
    ends = offsets + sizes
    
    # Reshape for broadcasting to compare all pairs: (N, 1, 2) vs (1, N, 2)
    offsets_i = tf.expand_dims(offsets, 1)
    ends_i = tf.expand_dims(ends, 1)
    offsets_j = tf.expand_dims(offsets, 0)
    ends_j = tf.expand_dims(ends, 0)
    
    # Check overlap condition for intervals: (start_i < end_j) AND (start_j < end_i)
    # This must hold true for ALL dimensions (2 here) for shards to overlap
    overlaps = tf.logical_and(
        offsets_i < ends_j,
        offsets_j < ends_i
    )
    
    # Reduce across dimensions (axis 2) to check if all dims overlap
    all_dims_overlap = tf.reduce_all(overlaps, axis=2)
    
    # Create a mask for the upper triangle (i < j) to ignore self-comparison and duplicates
    num_shards = tf.shape(shards)[0]
    # band_part(..., -1, 0) returns lower triangle including diagonal
    lower_tri_incl_diag = tf.linalg.band_part(tf.ones([num_shards, num_shards]), -1, 0)
    # We want the upper triangle (i < j), so we take the logical NOT of the lower triangle
    mask = tf.logical_not(tf.cast(lower_tri_incl_diag, tf.bool))
    
    # Apply mask and check for any overlaps
    valid_overlaps = tf.logical_and(all_dims_overlap, mask)
    
    if tf.reduce_any(valid_overlaps):
        raise ValueError("Overlapping shards detected")

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