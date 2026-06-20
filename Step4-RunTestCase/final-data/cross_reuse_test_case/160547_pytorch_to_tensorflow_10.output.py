import tensorflow as tf
from collections import namedtuple

def test_namedtuple_spatial_dropout():
    """
    Adapted test case for tf.keras.layers.SpatialDropout1D based on 
    PyTorch issue 160547 regarding NamedTuple inputs.
    """
    # Define a NamedTuple to hold inputs and the training flag
    LayerInput = namedtuple('LayerInput', 'inputs training')
    
    # Initialize the layer
    layer = tf.keras.layers.SpatialDropout1D(rate=0.5)
    
    # Create input data (3D tensor: samples, timesteps, channels)
    # SpatialDropout1D expects a 3D input.
    data = tf.ones((2, 10, 5))
    
    # Pack inputs into NamedTuple
    inp = LayerInput(data, True)
    
    # Test 1: Call with unpacked NamedTuple (Training mode)
    # In the original bug, strict=False failed. Here we verify if the layer
    # handles the unpacked NamedTuple correctly in training mode.
    print("Testing with NamedTuple unpacking (Training=True)...")
    try:
        output_train = layer(*inp)
        assert output_train.shape == data.shape, "Shape mismatch in training mode"
        print(f"Success. Output shape: {output_train.shape}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: Call with unpacked NamedTuple (Inference mode)
    # Corresponds to the 'strict=True' or alternative execution path.
    inp_inf = LayerInput(data, False)
    print("Testing with NamedTuple unpacking (Training=False)...")
    try:
        output_inf = layer(*inp_inf)
        assert output_inf.shape == data.shape, "Shape mismatch in inference mode"
        print(f"Success. Output shape: {output_inf.shape}")
    except Exception as e:
        print(f"Error: {e}")

    # Workaround: Convert to kwargs (as suggested in the original bug report)
    # This verifies that the layer works when arguments are passed explicitly.
    inp_kwargs = {'inputs': data, 'training': False}
    print("Testing with kwargs workaround...")
    output_kwargs = layer(**inp_kwargs)
    assert output_kwargs.shape == data.shape, "Shape mismatch with kwargs"
    print(f"Success. Output shape: {output_kwargs.shape}")

if __name__ == "__main__":
    test_namedtuple_spatial_dropout()