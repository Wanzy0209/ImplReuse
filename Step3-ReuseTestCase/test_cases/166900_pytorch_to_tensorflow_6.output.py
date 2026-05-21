import torch
import tensorflow as tf

class Foo:
    pass

class Bar:
    def __eq__(self, other):
        return super().__eq__(other)

    def __hash__(self):
        return 0

# We use tf.function to mimic the compilation/tracing behavior of torch.compile
@tf.function
def fn(x, obj):
    # We use the target API tf.compat.v1.name_scope
    with tf.compat.v1.name_scope("custom_scope"):
        # Replicate the logic that caused the bug in PyTorch:
        # Mutating an object attribute with a custom class instance inside a compiled scope
        obj.attr = {3: Bar()}
        return x + 1

if __name__ == "__main__":
    try:
        fn(tf.ones(3), Foo())
        print("Test executed successfully.")
    except Exception as e:
        print(f"Test failed with error: {e}")