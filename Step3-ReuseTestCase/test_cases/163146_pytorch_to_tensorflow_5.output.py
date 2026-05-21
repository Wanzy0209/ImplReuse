import torch
import tensorflow as tf
import numpy as np

def test_clone_model_with_dynamic_slice():
    """
    Adapts the PyTorch dynamic slicing bug reproduction to TensorFlow.
    
    Original Issue: torch.export.export fails with a "Data dependent error" 
    when encountering a slice operation where the stop index is a tensor 
    (e.g., x[:, :max_item_num, :]).

    Target API: tf.keras.models.clone_model
    Context: tf.keras.models.clone_model creates a copy of the model architecture.
    Unlike torch.export.export, it does not trace the execution graph with specific inputs.
    Therefore, it should handle models containing dynamic slicing logic without 
    raising data-dependent tracing errors, as it only copies the structure.
    """

    # Define a model that mimics the logic in the bug report.
    # The original PyTorch code snippet causing the error was:
    # selected_item_embedding = item_embedding[:, :max_item_num, :]
    class Mlp(tf.keras.Model):
        def __init__(self):
            super(Mlp, self).__init__()
            # Adding a layer to ensure it is a valid Keras model
            self.dense = tf.keras.layers.Dense(64)

        def call(self, inputs):
            # inputs is a tuple: (item_embedding, max_item_num)
            item_embedding, max_item_num = inputs
            
            # Reproduce the core logic: Dynamic slicing based on a tensor value.
            # In TensorFlow, this is a standard operation (tf.strided_slice).
            # item_embedding shape: [batch, seq_len, dim]
            # max_item_num shape: [] (scalar)
            selected_item_embedding = item_embedding[:, :max_item_num, :]
            
            return self.dense(selected_item_embedding)

    # 1. Instantiate the original model
    model = Mlp()

    # 2. Build the model by calling it once with dummy data
    # Shapes mimic the PyTorch error log: item_embedding [s10, s64, 64]
    # We use concrete values here for the dummy call.
    batch_size = 2
    seq_len = 200
    dim = 64
    dummy_embedding = tf.random.normal((batch_size, seq_len, dim))
    dummy_max_item_num = tf.constant(100, dtype=tf.int32)
    
    # Run a forward pass to build the layer weights
    _ = model((dummy_embedding, dummy_max_item_num))

    # 3. Test the Similar API: tf.keras.models.clone_model
    # This API clones the architecture. It does not execute the graph.
    # We expect this to succeed because it doesn't evaluate the tensor values 
    # inside the slicing operation during the cloning process.
    try:
        cloned_model = tf.keras.models.clone_model(model)
        
        # Verify the cloned model is functional and preserves the logic
        # Note: clone_model does not copy weights, so we initialize them or just check structure.
        # We can run a forward pass to ensure the graph structure is valid.
        # Since weights are new, output values will differ, but shape should match.
        output = cloned_model((dummy_embedding, dummy_max_item_num))
        
        # Assertions
        assert isinstance(cloned_model, tf.keras.Model), "Cloned object is not a Keras Model"
        assert output.shape == (batch_size, 100, 64), f"Unexpected output shape: {output.shape}"
        
        print("Test Passed: tf.keras.models.clone_model successfully handled the model with dynamic slicing.")
        
    except Exception as e:
        print(f"Test Failed: {e}")
        raise

if __name__ == "__main__":
    test_clone_model_with_dynamic_slice()