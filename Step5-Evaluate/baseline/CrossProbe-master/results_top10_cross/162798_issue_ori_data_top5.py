```python
import tensorflow as tf

# Conversion: Define nanmedian helper as TF does not have a direct equivalent
def nanmedian(x):
    # Filter out NaN values
    mask = ~tf.math.is_nan(x)
    clean_values = tf.boolean_mask(x, mask)

    # Handle empty tensor case (PyTorch returns nan for empty input)
    # tf.math.reduce_median raises error on empty tensor, so we use tf.cond
    is_empty = tf.equal(tf.size(clean_values), 0)

    def compute_median():
        return tf.math.reduce_median(clean_values)

    def return_nan():
        return tf.constant(float('nan'), dtype=x.dtype)

    return tf.cond(is_empty, return_nan, compute_median)

# Conversion: torch.empty((0,), dtype=torch.float32) -> tf.constant
x = tf.constant([], dtype=tf.float32)
print("Input tensor:", x)

# nanmedian on CPU
cpu_result = nanmedian(x)
print("CPU result:", cpu_result) # tensor(nan)

# nanmedian on CUDA
# Conversion: torch.cuda.is_available() -> tf.config.list_physical_devices('GPU')
if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        # Conversion: x.to('cuda') -> explicit placement via identity
        cuda_x = tf.identity(x)
        cuda_result = nanmedian(cuda_x)
        print("CUDA result:", cuda_result) # tensor(nan, device='/GPU:0')

# nanmedian on MPS
# Conversion: torch.backends.mps.is_available() -> No direct MPS API in TF.
# On macOS, Metal acceleration is handled via the GPU device.
# We check for GPU availability to preserve structure.
if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        mps_x = tf.identity(x)
        mps_result = nanmedian(mps_x)
        # Note: TF returns nan for empty input, whereas PyTorch MPS might return 0
        print("MPS result:", mps_result) # tensor(nan, device='/GPU:0')
```