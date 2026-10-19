```python
import tensorflow as tf

# Conversion: torch.utils.data.Dataset is a base class in PyTorch.
# In TensorFlow, we typically use tf.data.Dataset directly, but to preserve structure
# we define a class to hold the data.
class TestDataset: # basic Dataset for testing
        def __init__(self, x: tf.Tensor, y: tf.Tensor):
            # super().__init__() # Not strictly required in Python for simple classes
            self.x = x
            self.y = y

        def __len__(self):
            return tf.shape(self.x)[0]

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]


def test_bug(device: str = 'cuda'):
    # Conversion: torch.set_default_device sets a global default.
    # TensorFlow uses context managers for device placement.
    # Mapping 'cuda' to '/GPU:0' and 'cpu' to '/CPU:0' for compatibility.
    tf_device = '/GPU:0' if device == 'cuda' else '/CPU:0'
    
    with tf.device(tf_device): # if device is 'cuda' then operations run on GPU
        # Conversion: torch.randn -> tf.random.normal
        x = tf.random.normal((100, 3))
        y = tf.random.normal((100, 2))

        dataset = TestDataset(x, y)

        # Conversion: torch.utils.data.random_split
        # TF does not have a direct random_split function. 
        # We create a Dataset, shuffle it (to mimic random_split's randomness),
        # and then use take/skip to partition.
        
        # Create a tf.data.Dataset from the tensors
        full_dataset = tf.data.Dataset.from_tensor_slices((dataset.x, dataset.y))
        
        # Shuffle to mimic the random nature of random_split
        # Using a fixed seed to match the commented out generator in source
        full_dataset = full_dataset.shuffle(100, seed=42)

        total = 100
        train_len = int(0.7 * total)
        val_len = int(0.2 * total)
        
        train_dataset = full_dataset.take(train_len)
        val_dataset = full_dataset.skip(train_len).take(val_len)
        test_dataset = full_dataset.skip(train_len + val_len)

        print(f"Device {device} worked.")

test_bug(device='cpu') # works
test_bug(device='cuda') # works in TF (if GPU available)
```