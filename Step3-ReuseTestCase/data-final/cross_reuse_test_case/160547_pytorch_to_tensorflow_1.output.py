import torch
import tensorflow as tf
from collections import namedtuple

def test_namedtuple():
    # Define the NamedTuple structure
    Point = namedtuple('Point', ['x', 'y'])

    # Define a computation that takes two arguments
    # Corresponds to PyTorch's M.forward(self, x, y)
    def computation(x, y):
        return x + y

    # Create input using NamedTuple
    inp = Point(tf.ones(3), tf.ones(3))

    # Verify direct execution works (PyTorch: M()(*inp))
    print("Direct execution:", computation(*inp))

    # Test 1: Compile with NamedTuple input
    # PyTorch: torch.export.export(M(), inp, strict=False)
    # TF: tf.xla.experimental.compile(computation, inputs=inp)
    # Note: tf.xla.experimental.compile does not have a 'strict' parameter.
    # We verify if the API handles the NamedTuple (which is a tuple subclass) correctly.
    try:
        # Since Point is a tuple, compile should unpack it into x and y
        ep = tf.xla.experimental.compile(computation, inputs=inp)
        print("XLA Compile (unpacked args): Success")
        print(ep)
    except Exception as e:
        print(f"XLA Compile (unpacked args) Failed: {e}")

    # Test 2: Workaround / Alternative (Function accepts the container)
    # PyTorch workaround: convert to kwargs and change signature to forward(self, **kwargs).
    # TF equivalent: Define function to accept the namedtuple structure directly.
    def computation_container(pt):
        return pt.x + pt.y

    try:
        ep = tf.xla.experimental.compile(computation_container, inputs=inp)
        print("XLA Compile (container arg): Success")
        print(ep)
    except Exception as e:
        print(f"XLA Compile (container arg) Failed: {e}")

if __name__ == "__main__":
    test_namedtuple()