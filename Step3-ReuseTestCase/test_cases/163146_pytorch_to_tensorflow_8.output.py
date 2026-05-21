import torch
import tensorflow as tf
import numpy as np

def test_rnn_dynamic_slice_export():
    """
    Adapted test case from PyTorch Issue #163146.
    
    Original Bug: torch.export.export fails with a "Data dependent error" 
    when encountering a dynamic slice operation (e.g., tensor[:, :dynamic_val, :])
    where the slice index is a runtime Tensor rather than a constant.
    
    This test verifies the behavior of tf.keras.layers.RNN when used within
    a graph context (tf.function) involving similar dynamic slicing logic.
    """
    
    class DynamicSliceRNN(tf.keras.Model):
        def __init__(self):
            super(DynamicSliceRNN, self).__init__()
            # Initialize RNN layer with a SimpleRNNCell
            self.rnn = tf.keras.layers.RNN(
                cell=tf.keras.layers.SimpleRNNCell(units=10),
                return_sequences=False,
                return_state=False
            )

        @tf.function
        def call(self, item_embedding, max_item_num):
            """
            Mimics the logic from the PyTorch bug report:
            selected_item_embedding = item_embedding[:, :max_item_num, :]
            
            Args:
                item_embedding: Input tensor of shape (batch, time, features)
                max_item_num: Scalar Tensor determining the slice length.
            """
            # Perform dynamic slicing based on the input tensor max_item_num
            # This corresponds to the line causing the error in PyTorch:
            # selected_item_embedding = item_embedding[:, :max_item_num, :]
            selected_item_embedding = item_embedding[:, :max_item_num, :]
            
            # Pass the dynamically sliced tensor to the RNN layer
            output = self.rnn(selected_item_embedding)
            return output

    # 1. Setup test data
    # Shape: (Batch=2, Time=20, Features=64)
    batch_size = 2
    max_time = 20
    features = 64
    
    # Create a random tensor
    item_embedding = tf.random.normal((batch_size, max_time, features))
    
    # Create a scalar tensor for the slice limit.
    # In the PyTorch bug, this was a Tensor(shape: torch.Size([]), stride: (), storage_offset: 0)
    # We use int32 as is standard for indexing/slicing dimensions in TF.
    max_item_num = tf.constant(10, dtype=tf.int32)

    # 2. Instantiate the model
    model = DynamicSliceRNN()

    # 3. Run the test
    # We expect this to run successfully. In the PyTorch bug, torch.export.export
    # failed because it could not evaluate the dynamic value 'u1' (max_item_num)
    # during the export process.
    try:
        # Calling a tf.function triggers the tracing process (similar to export)
        result = model(item_embedding, max_item_num)
        
        # Verify output shape
        # Batch size should be preserved, RNN units should be the last dimension
        expected_shape = (batch_size, 10)
        assert result.shape == expected_shape, \
            f"Expected output shape {expected_shape}, but got {result.shape}"
            
        print("Test Passed: tf.keras.layers.RNN handled dynamic slicing within graph context successfully.")
        
    except Exception as e:
        print(f"Test Failed: The model raised an error during execution/tracing.")
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    test_rnn_dynamic_slice_export()