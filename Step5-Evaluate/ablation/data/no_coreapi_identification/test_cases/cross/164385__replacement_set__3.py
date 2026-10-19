import tensorflow as tf
import numpy as np

"""
Test case adapted from Issue 164385: FloorDiv generate sympy rational
Replacements applied based on API similarity:
- sympy.Symbol -> tf.experimental.numpy.diagonal
- FloorDiv -> tf.experimental.numpy.diagonal
- sympy.srepr -> tf.experimental.numpy.diagonal
"""

print("Testing with tf.experimental.numpy.diagonal...")

# Create symbolic variables
# Original: s14 = sympy.Symbol('s14', integer=True, positive=True)
# Replacement: Use tf.experimental.numpy.diagonal to generate tensor inputs
# We create diagonal matrices to extract the "variables" (vectors)
s14 = tf.experimental.numpy.diagonal(tf.constant([[14, 0], [0, 14]], dtype=tf.int32))
s37 = tf.experimental.numpy.diagonal(tf.constant([[37, 0], [0, 37]], dtype=tf.int32))
s46 = tf.experimental.numpy.diagonal(tf.constant([[46, 0], [0, 46]], dtype=tf.int32))

# Build the numerator expression step by step
# Original: inner_expr = FloorDiv(s14 , 2016)
# Replacement: Construct a 2D tensor from s14 and 2016, then extract diagonal
# To make the operation runnable, we stack the inputs to form a matrix
inner_matrix = tf.stack([s14, [2016, 2016]])
inner_expr = tf.experimental.numpy.diagonal(inner_matrix)

middle_expr = (24 * s37 + 672) * inner_expr
numerator = middle_expr + 21
denominator = 22

print(f"Numerator: {numerator}")
print(f"Denominator: {denominator}")

# Create the FloorDiv expression
# Original: result = FloorDiv(numerator, denominator)
# Replacement: Construct a 2D tensor from numerator and denominator, then extract diagonal
result_matrix = tf.stack([numerator, [denominator, denominator]])
result = tf.experimental.numpy.diagonal(result_matrix)

print(f"FloorDiv result: {result}")

# Original: print(f"FloorDiv result: {sympy.srepr(result)}")
# Replacement: Use tf.experimental.numpy.diagonal on a diagonal matrix of the result
# This mimics the structural inspection of the result
srepr_result = tf.experimental.numpy.diagonal(tf.linalg.diag(result))
print(f"FloorDiv result: {srepr_result}")

# Assertion to verify the operation executed and returned a Tensor
assert isinstance(result, tf.Tensor), "Result should be a TensorFlow Tensor"