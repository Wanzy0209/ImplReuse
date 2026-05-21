import tensorflow as tf
import numpy as np

def test_categorical_hinge_precision_divergence():
    """
    Test case adapted from Issue 164086.
    
    The original bug report highlights an IncompatibleTypeError between pointer<fp16> 
    and triton.language.float64 during compilation, triggered by mixed precision 
    operations (int64 inputs, float16 tensors).
    
    This test verifies that tf.keras.losses.categorical_hinge handles similar 
    mixed precision scenarios (float64/int64 inputs with float16 predictions) 
    correctly without raising type errors or producing numerical instability (NaNs).
    """
    
    batch_size = 4
    num_classes = 3

    # Scenario 1: y_true is float64 (double), y_pred is float16 (half)
    # This directly addresses the "fp16 and float64" incompatibility mentioned in the bug title.
    y_true_f64 = tf.constant(np.random.randint(0, 2, size=(batch_size, num_classes)), dtype=tf.float64)
    y_pred_fp16 = tf.random.uniform((batch_size, num_classes), dtype=tf.float16)

    # The API implementation casts y_true to y_pred.dtype (float16) internally.
    # We verify this operation completes successfully and maintains numerical stability.
    try:
        loss = tf.keras.losses.categorical_hinge(y_true_f64, y_pred_fp16)
        # Verify output is valid (no NaNs from precision underflow/overflow during cast)
        assert not tf.reduce_any(tf.math.is_nan(loss)), "Loss produced NaNs with float64/float16 inputs"
        assert loss.dtype == tf.float16, "Loss output dtype should match y_pred"
    except (TypeError, tf.errors.InvalidArgumentError) as e:
        print(f"Type error encountered: {e}")
        raise

    # Scenario 2: y_true is int64, y_pred is float16
    # Mirroring the 't0' (int64) usage in the original bug report where int64 tensors
    # interact with float16 computation graphs.
    y_true_int64 = tf.constant(np.random.randint(0, 2, size=(batch_size, num_classes)), dtype=tf.int64)
    y_pred_fp16_2 = tf.random.uniform((batch_size, num_classes), dtype=tf.float16)

    try:
        loss_int = tf.keras.losses.categorical_hinge(y_true_int64, y_pred_fp16_2)
        assert not tf.reduce_any(tf.math.is_nan(loss_int)), "Loss produced NaNs with int64/float16 inputs"
        assert loss_int.dtype == tf.float16
    except (TypeError, tf.errors.InvalidArgumentError) as e:
        print(f"Type error encountered: {e}")
        raise

    print("Test Passed: Precision handling is robust for mixed types.")

if __name__ == '__main__':
    test_categorical_hinge_precision_divergence()