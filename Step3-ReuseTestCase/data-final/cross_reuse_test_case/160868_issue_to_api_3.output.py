import torch
import tensorflow as tf
import numpy as np
import os

# Disable GPU to match the original test's intent of minimizing noise
os.environ["CUDA_VISIBLE_DEVICES"] = ""

class LogCoshHugeValueTest:
    """
    Test case for tf.keras.metrics.LogCoshError adapted from the 
    torch.slice_copy huge step bug report.
    
    Original Bug Logic:
    - Input: A huge step value (2**63 - 1).
    - Operation: slice_copy -> reciprocal.
    - Expected: Empty tensor / No crash.
    - Actual (Bug): Segfault in Inductor.
    
    Adapted Logic for LogCoshError:
    - Input: Huge values in y_pred/y_true (2**63 - 1) and empty inputs.
    - Operation: LogCoshError metric update.
    - Expected: Numerical result (inf or large) or 0 for empty, but no crash.
    """
    def __init__(self, huge_val=(2**63 - 1)):
        self.huge_val = huge_val
        self.metric = tf.keras.metrics.LogCoshError()

    def run_huge_value_test(self):
        print("[LogCoshError] Testing with huge input values...")
        # Mimic the 'huge step' by using a huge value in the input tensor
        y_true = tf.constant([0.0, 0.0], dtype=tf.float32)
        y_pred = tf.constant([float(self.huge_val), 0.0], dtype=tf.float32)
        
        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()
        
        # We expect inf or a large number due to log(cosh(huge)), but not a crash
        print(f"[LogCoshError] OK with huge value. Result: {result.numpy()}")
        assert not np.isnan(result.numpy()), "Result should not be NaN"
        # Reset for next test
        self.metric.reset_states()

    def run_empty_input_test(self):
        print("[LogCoshError] Testing with empty inputs...")
        # Mimic the 'empty tensor' result of the huge step slice
        y_true = tf.constant([], dtype=tf.float32)
        y_pred = tf.constant([], dtype=tf.float32)
        
        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()
        
        print(f"[LogCoshError] OK with empty inputs. Result: {result.numpy()}")
        # Result should be 0.0 for empty inputs in MeanMetricWrapper
        assert result.numpy() == 0.0, "Result should be 0.0 for empty inputs"

if __name__ == "__main__":
    test_suite = LogCoshHugeValueTest()
    
    # Run tests mirroring the original bug's conditions (extreme parameters)
    test_suite.run_huge_value_test()
    test_suite.run_empty_input_test()
    
    print("\nAll tests passed.")