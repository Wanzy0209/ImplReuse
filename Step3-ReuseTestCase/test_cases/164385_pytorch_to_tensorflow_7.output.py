import tensorflow as tf
import numpy as np

def test_sparse_concat_integer_preservation():
    """
    Adapted test case for tf.sparse.concat based on the FloorDiv bug report.
    
    Original Issue: FloorDiv((24*s37 + 672)*(((s14*s46)//2016)) + 21, 22) 
    was generating a sympy Rational (Mul) instead of preserving FloorDiv semantics.
    
    Adaptation: We test tf.sparse.concat with integer sparse tensors to ensure 
    it preserves the SparseTensor structure and integer dtypes, rather than 
    converting to dense or floating point types (analogous to the type coercion 
    in the original bug).
    """
    print("Testing tf.sparse.concat with complex integer inputs...")

    # Create sparse tensors with values corresponding to the constants in the original expression
    # (24, 672, 2016, 21, 22)
    # We use integer types to match the 'integer=True' constraint in the original bug report.
    
    # Tensor 1: Represents (24*s37 + 672)
    indices_1 = [[0, 0], [0, 1]]
    values_1 = [24, 672]
    shape_1 = [1, 3]
    st_1 = tf.sparse.SparseTensor(indices=indices_1, values=values_1, dense_shape=shape_1)

    # Tensor 2: Represents (s14*s46)//2016 and +21
    indices_2 = [[0, 0], [0, 1]]
    values_2 = [2016, 21]
    shape_2 = [1, 3]
    st_2 = tf.sparse.SparseTensor(indices=indices_2, values=values_2, dense_shape=shape_2)

    # Tensor 3: Represents the denominator 22
    indices_3 = [[0, 0]]
    values_3 = [22]
    shape_3 = [1, 3]
    st_3 = tf.sparse.SparseTensor(indices=indices_3, values=values_3, dense_shape=shape_3)

    # Perform the concatenation (analogous to building the complex expression)
    # Concatenating along axis 0
    result = tf.sparse.concat(sp_inputs=[st_1, st_2, st_3], axis=0)

    print(f"Result: {result}")
    print(f"Result type: {type(result)}")
    print(f"Result dtype: {result.dtype}")

    # --- Assertions ---

    # 1. Check that the result is a SparseTensor (not converted to dense or other types)
    # This mirrors the check in the original bug where the result type changed from FloorDiv to Mul.
    assert isinstance(result, tf.sparse.SparseTensor), \
        f"Expected result to be tf.sparse.SparseTensor, but got {type(result)}"

    # 2. Check that the dtype is integer (not float/rational), preserving the integer nature of the inputs
    # The original bug was FloorDiv -> Rational (loss of integer semantics).
    # Here we check if integer sparse tensors remain integer.
    assert result.dtype.is_integer, \
        f"Expected integer dtype, but got {result.dtype}"

    # 3. Verify the values are correct and preserved
    expected_values = [24, 672, 2016, 21, 22]
    result_values = result.values.numpy()
    
    assert np.array_equal(result_values, expected_values), \
        f"Values mismatch: expected {expected_values}, got {list(result_values)}"

    print("Test passed: tf.sparse.concat preserves sparse structure and integer types.")

if __name__ == "__main__":
    test_sparse_concat_integer_preservation()