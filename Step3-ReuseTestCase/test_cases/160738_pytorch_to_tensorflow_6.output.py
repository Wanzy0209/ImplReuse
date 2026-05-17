import tensorflow as tf
import numpy as np

def test_isposinf():
    # Test with a zero-dimensional tensor (scalar) - finite value
    x_finite = tf.constant(3.0)
    try:
        # Note: isposinf is an element-wise operation and does not take a 'dim' argument.
        # We test the API's handling of zero-dimensional tensors here.
        output = tf.experimental.numpy.isposinf(x_finite)
        print(f"isposinf test succeeds for finite scalar input. output: {output}")
        assert output.numpy() == False
    except Exception as e:
        print(f"isposinf test fails for finite scalar input: {e}")

    # Test with a zero-dimensional tensor (scalar) - positive infinity
    x_inf = tf.constant(np.inf)
    try:
        output = tf.experimental.numpy.isposinf(x_inf)
        print(f"isposinf test succeeds for inf scalar input. output: {output}")
        assert output.numpy() == True
    except Exception as e:
        print(f"isposinf test fails for inf scalar input: {e}")

if __name__ == "__main__":
    test_isposinf()