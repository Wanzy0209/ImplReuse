import tensorflow as tf
import numpy as np

def test_cosine_similarity_edge_cases():
    """
    Test case for tf.keras.losses.CosineSimilarity.
    
    This test is derived from the similarity to Issue 160980, where a specific
    configuration (SM_75 architecture) caused a build failure due to undefined 
    references. 
    
    For the CosineSimilarity API, the "specific configuration" analogous to an 
    edge case is the presence of zero vectors. The API documentation states:
    "If either `y_true` or `y_pred` is a zero vector, cosine similarity will be 0".
    
    This test verifies that the API correctly handles this specific edge case,
    ensuring the implementation is robust for its defined special conditions,
    just as the build system should be robust for specific architectures.
    """
    
    # Initialize the loss function with axis=1 as per the API documentation
    cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)

    # Case 1: Standard inputs (Baseline behavior)
    # y_true = [[1., 1.], [1., 1.]]
    # y_pred = [[1., 1.], [1., 1.]]
    # l2_norm(y_true) = [[0.707, 0.707], [0.707, 0.707]]
    # l2_norm(y_pred) = [[0.707, 0.707], [0.707, 0.707]]
    # Dot product = 1.0 for both rows.
    # Loss = -1.0 (sum over batch size / 2)
    y_true = [[1., 1.], [1., 1.]]
    y_pred = [[1., 1.], [1., 1.]]
    result = cosine_loss(y_true, y_pred)
    assert np.isclose(result.numpy(), -1.0), f"Expected -1.0 for identical vectors, got {result.numpy()}"

    # Case 2: Zero vector inputs (The "Edge Case")
    # This mirrors the specific SM_75 configuration in the bug report.
    # y_true contains a zero vector in the first row.
    # y_pred contains a non-zero vector in the first row.
    # Expected behavior: Similarity is 0 for the first row.
    y_true_zero = [[0., 0.], [1., 1.]]
    y_pred_zero = [[1., 1.], [1., 1.]]
    
    # Row 1: [0,0] vs [0.707, 0.707] -> Similarity 0 -> Loss 0
    # Row 2: [0.707, 0.707] vs [0.707, 0.707] -> Similarity 1 -> Loss -1
    # Average Loss = (0 + -1) / 2 = -0.5
    result_zero = cosine_loss(y_true_zero, y_pred_zero)
    assert np.isclose(result_zero.numpy(), -0.5), f"Expected -0.5 with zero vector, got {result_zero.numpy()}"

    print("Test passed: CosineSimilarity handles zero vectors correctly.")

if __name__ == "__main__":
    test_cosine_similarity_edge_cases()