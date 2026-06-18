import torch
import tensorflow as tf

def test_serialize_compile():
    """
    Test case adapted from PyTorch issue 164823.
    Original issue: torch.compile fails with models using to_sparse()/to_dense().
    This test verifies the analogous behavior in TensorFlow using tf.io.serialize_tensor
    (format conversion) within a tf.function (compiled) context.
    """
    
    # Define a model that performs a format conversion roundtrip
    # analogous to to_sparse() -> to_dense()
    class TestModel(tf.Module):
        @tf.function
        def __call__(self, x):
            # Convert to serialized format (analogous to to_sparse)
            x_serialized = tf.io.serialize_tensor(x)
            
            # Convert back to tensor (analogous to to_dense)
            # Note: parse_tensor requires explicit dtype
            x_parsed = tf.io.parse_tensor(x_serialized, out_type=tf.float32)
            
            # Perform an operation on the converted data
            result = x_parsed * 2
            return result

    # Input data
    x = tf.random.normal((10, 10))

    model = TestModel()
    
    # 1. Eager execution (tf.function can be called eagerly, but we test the graph logic)
    # To strictly mimic the "eager vs compiled" split in PyTorch:
    # We run the logic without the decorator first, then with.
    
    def eager_logic(x):
        x_serialized = tf.io.serialize_tensor(x)
        x_parsed = tf.io.parse_tensor(x_serialized, out_type=tf.float32)
        return x_parsed * 2

    print("Eager output:", eager_logic(x))
    
    # 2. Compiled execution (tf.function is analogous to torch.compile)
    # The __call__ method is already decorated with @tf.function
    compiled_output = model(x)
    print("Compiled output:", compiled_output)

    # Assert that the compiled output matches the eager output
    # This ensures the compilation handled the serialization/parsing correctly
    assert tf.reduce_all(tf.equal(eager_logic(x), compiled_output)), "Mismatch between eager and compiled outputs"

if __name__ == "__main__":
    test_serialize_compile()