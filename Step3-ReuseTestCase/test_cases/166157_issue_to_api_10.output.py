import tensorflow as tf
import numpy as np

# This class mimics the behavior of the proposed OpenRegDeviceAllocator fix.
# It wraps a low-level operation (tf.math.exp) and adds statistics tracking,
# addressing the "black box" issue described in the bug report.
class TrackedExpOperator:
    def __init__(self):
        # Mimicking c10::CachingDeviceAllocator::DeviceStats
        self.stats = {
            'allocated_bytes': 0,
            'num_ops': 0
        }

    def compute(self, x):
        """
        Performs the exponential operation while tracking resource usage.
        This mirrors the 'allocate' method in OpenRegDeviceAllocator where
        stats_.allocated_bytes is increased.
        """
        # Calculate the "cost" (bytes) of the operation
        nbytes = x.numpy().nbytes
        
        # Perform the actual operation (similar to orMalloc)
        result = tf.math.exp(x)
        
        # Update statistics (The fix for the bug)
        self.stats['allocated_bytes'] += nbytes
        self.stats['num_ops'] += 1
        
        return result

    def get_stats(self):
        """Returns the accumulated statistics."""
        return self.stats

def test_tracked_exp_statistics():
    """
    Test case to verify that the wrapper correctly tracks statistics,
    ensuring the system is not a 'black box' as described in the issue.
    """
    tracker = TrackedExpOperator()
    
    # 1. Verify initial state (Bug: previously had no visibility)
    initial_stats = tracker.get_stats()
    assert initial_stats['allocated_bytes'] == 0, "Initial allocated bytes should be 0"
    assert initial_stats['num_ops'] == 0, "Initial operation count should be 0"
    
    # 2. Perform an operation using the similar API (tf.math.exp)
    input_tensor = tf.constant([1.0, 2.0, 3.0])
    _ = tracker.compute(input_tensor)
    
    # 3. Verify statistics are updated (Fix: migration to c10::DeviceAllocator interface)
    stats = tracker.get_stats()
    expected_bytes = input_tensor.numpy().nbytes
    
    assert stats['allocated_bytes'] == expected_bytes, \
        f"Expected {expected_bytes} bytes tracked, but got {stats['allocated_bytes']}"
    assert stats['num_ops'] == 1, \
        f"Expected 1 operation tracked, but got {stats['num_ops']}"
    
    # 4. Test accumulation
    _ = tracker.compute(input_tensor)
    stats = tracker.get_stats()
    assert stats['allocated_bytes'] == expected_bytes * 2, \
        "Statistics should accumulate across multiple calls"
    assert stats['num_ops'] == 2, \
        "Operation count should accumulate across multiple calls"

    print("Test passed: Statistics are correctly tracked and visible.")

if __name__ == "__main__":
    test_tracked_exp_statistics()