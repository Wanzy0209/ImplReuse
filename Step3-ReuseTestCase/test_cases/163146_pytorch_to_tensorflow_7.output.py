import torch
import tensorflow as tf
import numpy as np

def test_dynamic_slicing_with_keras_input():
    """
    Adapts the PyTorch dynamic slicing issue to TensorFlow's tf.keras.Input.
    
    Original Issue: torch.export.export fails with a "Data dependent error" 
    when slicing a tensor using another tensor as the stop index 
    (e.g., tensor[:, :dynamic_tensor, :]).
    
    This test verifies if defining a model with tf.keras.Input and performing
    similar dynamic slicing operations works in TensorFlow.
    """
    
    # 1. Define inputs using tf.keras.Input
    # Corresponds to 'item_embedding' in the PyTorch traceback
    # Shape: (Batch, Sequence, Dim) -> (None, None, 64)
    # We allow None for batch size and sequence length to support dynamic shapes.
    item_embedding = tf.keras.Input(
        shape=(None, 64), 
        batch_size=None, 
        name="item_embedding", 
        dtype=tf.float32
    )

    # Corresponds to 'max_item_num' in the PyTorch traceback
    # Shape: Scalar -> ()
    max_item_num = tf.keras.Input(
        shape=(), 
        name="max_item_num", 
        dtype=tf.int32
    )

    # 2. Reproduce the core logic: Data dependent slicing
    # PyTorch code: selected_item_embedding = item_embedding[:, :max_item_num, :]
    # TensorFlow supports dynamic slicing natively in the graph.
    # We attempt to slice the sequence dimension (dim 1) up to the value of max_item_num.
    try:
        selected_item_embedding = item_embedding[:, :max_item_num, :]
    except Exception as e:
        print(f"Error during graph construction: {e}")
        raise

    # 3. Construct the model
    # This step traces the graph, similar to what torch.export.export attempts.
    model = tf.keras.Model(
        inputs=[item_embedding, max_item_num], 
        outputs=selected_item_embedding
    )

    # 4. Verify with concrete data
    # Create dummy data
    # Batch size 2, sequence length 10, dim 64
    dummy_emb = np.random.rand(2, 10, 64).astype(np.float32)
    # Slice limit 5
    dummy_limit = np.array(5, dtype=np.int32)

    # Run the model (executes the traced graph)
    output = model.predict([dummy_emb, dummy_limit], verbose=0)

    # Assertions
    # Expected shape: (2, 5, 64) - Batch 2, Sliced to 5, Dim 64
    expected_shape = (2, 5, 64)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    # Verify content correctness
    assert np.allclose(output[0], dummy_emb[0, :5, :]), "Sliced data does not match input data"

    print("Test passed: Dynamic slicing with tf.keras.Input works as expected.")

if __name__ == "__main__":
    test_dynamic_slicing_with_keras_input()