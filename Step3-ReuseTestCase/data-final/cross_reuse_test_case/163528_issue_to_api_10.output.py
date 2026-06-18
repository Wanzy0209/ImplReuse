import tensorflow as tf
import numpy as np

# Replicating the logic from the Similar API information provided
def top_k_categorical_accuracy(y_true, y_pred, k=5):
    """Computes how often targets are in the top `K` predictions."""
    return tf.cast(
        tf.nn.in_top_k(
            y_pred, tf.argmax(y_true, axis=-1), k), tf.floatx())

class TopKModel(tf.Module):
    def __init__(self, k=5):
        super().__init__()
        self.k = k

    @tf.function
    def call_compiled(self, y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, self.k)

    def call_eager(self, y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, self.k)

def test_device(device_name, y_true, y_pred):
    with tf.device(device_name):
        model = TopKModel(k=5)
        
        # warm up
        y_original = model.call_eager(y_true, y_pred)
        y_compiled = model.call_compiled(y_true, y_pred)

        # proper inference
        y_original = model.call_eager(y_true, y_pred)
        y_compiled = model.call_compiled(y_true, y_pred)

        diff = tf.reduce_max(tf.abs(y_original - y_compiled)).numpy()
        print(f'device: {device_name}, diff: {diff}')
        print('original', y_original[:5])
        print('compiled', y_compiled[:5])
        
        # Assert to check if the bug (compilation mismatch) exists in the similar API
        assert np.allclose(y_original.numpy(), y_compiled.numpy()), \
            f"Bug detected: Compiled results differ from eager on {device_name}"

def main():
    batch_size = 32
    num_classes = 10
    np.random.seed(42)
    
    # Generate dummy data
    y_true_indices = np.random.randint(0, num_classes, size=(batch_size,))
    y_true = tf.one_hot(y_true_indices, depth=num_classes)
    y_pred = np.random.rand(batch_size, num_classes).astype(np.float32)

    # Test on CPU
    test_device('/cpu:0', y_true, y_pred)

    # Test on GPU if available
    if tf.config.list_physical_devices('GPU'):
        test_device('/gpu:0', y_true, y_pred)
    else:
        print("GPU not available, skipping GPU test.")

if __name__ == "__main__":
    main()