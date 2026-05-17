import torch
import tensorflow as tf

def test_dropout():
    # Setup input tensor
    x = tf.ones([2, 3])
    
    # Create a scalar (zero-dimensional) tensor for noise_shape
    # This mirrors the 'index = torch.tensor(1)' from the PyTorch bug
    noise_shape = tf.constant(1)

    try:
        # Attempt to run dropout with scalar noise_shape
        # Note: tf.keras.random.dropout requires a seed
        output = tf.keras.random.dropout(x, rate=0.5, noise_shape=noise_shape, seed=42)
        print(f"dropout test succeeds. output shape: {output.shape}")
    except Exception as e:
        print(f"dropout test fails: {e}")

if __name__ == "__main__":
    test_dropout()