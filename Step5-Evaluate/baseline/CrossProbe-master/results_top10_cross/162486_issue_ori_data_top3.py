```python
import tensorflow as tf

# In TensorFlow, we typically use tf.data.Dataset.from_tensor_slices
# instead of subclassing a Dataset class for simple tensor data.
# The logic below translates the TestDataset functionality.

def test_bug(device: str = 'cuda'):
    # Map PyTorch device string to TensorFlow device string
    # PyTorch: set_default_device(device)
    # TensorFlow: use context manager tf.device
    tf_device = '/GPU:0' if device == 'cuda' else '/CPU:0'
    
    with tf.device(tf_device):
        # torch.randn(100, 3) -> tf.random.normal
        x = tf.random.normal((100, 3))
        y = tf.random.normal((100, 2))

        # TestDataset(x, y) -> tf.data.Dataset.from_tensor_slices
        dataset = tf.data.Dataset.from_tensor_slices((x, y))

        # PyTorch: random_split(dataset, [0.7, 0.2, 0.1])
        # TensorFlow: shuffle then take/skip
        # Note: random_split in PyTorch handles fractions summing to 1.0
        # We calculate lengths explicitly here.
        total_size = 100
        train_size = int(0.7 * total_size)
        val_size = int(0.2 * total_size)
        
        # Shuffle to ensure random split (mimics generator behavior)
        dataset = dataset.shuffle(buffer_size=total_size, seed=42)

        train_dataset = dataset.take(train_size)
        remaining = dataset.skip(train_size)
        
        val_dataset = remaining.take(val_size)
        test_dataset = remaining.skip(val_size)

    print(f"Device {device} worked.")

test_bug(device='cpu') # works
# Check for GPU availability to avoid runtime error if no GPU is present
if tf.config.list_physical_devices('GPU'):
    test_bug(device='cuda') # works in TF (unlike the PyTorch bug context)
else:
    print("CUDA device not found, skipping GPU test.")
```