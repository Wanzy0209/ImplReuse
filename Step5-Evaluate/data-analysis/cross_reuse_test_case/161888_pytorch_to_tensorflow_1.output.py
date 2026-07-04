import tensorflow as tf
import numpy as np

print(tf.__version__)

# Recreate tensors similar to the PyTorch bug report
# tensor1: int16, shape (9, 3, 7)
tensor1 = tf.constant(np.random.randint(low=-100, high=100, size=(9, 3, 7)), dtype=tf.int16)

# tensor2: bool, shape (1, 6, 4, 8)
tensor2 = tf.constant(np.random.randint(low=0, high=2, size=(1, 6, 4, 8)), dtype=tf.bool)

# The malicious integer from the original bug
huge_int = 154691921484029491302139942063978250367

# Attempt to call the API with the tensors and the malicious integer
# We pass the huge integer as a keyword argument to test robustness against invalid inputs
try:
    result = tf.compat.v1.losses.sparse_softmax_cross_entropy(
        tensor1,
        tensor2,
        **{huge_int: ()}
    )
    print("Result:", result)
except Exception as e:
    print(f"Exception: {e}")