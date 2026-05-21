import torch
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.training import queue_runner_impl

def test_queue_runner_collection_consistency():
    """
    Adapted test case for Issue 166012: tlparse entries on cache hit and not are inconsistent.
    
    Original Context: torch.compile generated inconsistent log entries between cache miss and cache hit.
    Adapted Context: tf.compat.v1.train.add_queue_runner adds entries to a graph collection.
    This test verifies that the collection entries (analogous to log entries) are consistent
    after multiple additions (simulating multiple runs/states).
    """
    
    # Reset the default graph to ensure a clean state for each test run
    tf.compat.v1.reset_default_graph()

    # Create a simple FIFO queue
    q = tf.compat.v1.FIFOQueue(capacity=10, dtypes=[tf.float32], shapes=[])
    
    # Create an enqueue operation
    enqueue_op = q.enqueue([1.0])
    
    # Create a QueueRunner
    qr = queue_runner_impl.QueueRunner(q, [enqueue_op])

    # --- Step 1: Simulate "Cache Miss" (First Addition) ---
    # Add the first queue runner to the default collection
    tf.compat.v1.train.add_queue_runner(qr)
    
    # Retrieve the collection state (analogous to checking logs/entries)
    collection_state_miss = ops.get_collection(ops.GraphKeys.QUEUE_RUNNERS)
    
    # --- Step 2: Simulate "Cache Hit" (Second Addition/State Check) ---
    # Create a second queue runner to simulate a subsequent operation
    q2 = tf.compat.v1.FIFOQueue(capacity=10, dtypes=[tf.float32], shapes=[])
    enqueue_op2 = q2.enqueue([2.0])
    qr2 = queue_runner_impl.QueueRunner(q2, [enqueue_op2])
    
    # Add the second queue runner
    tf.compat.v1.train.add_queue_runner(qr2)
    
    # Retrieve the collection state again
    collection_state_hit = ops.get_collection(ops.GraphKeys.QUEUE_RUNNERS)

    # --- Verification: Check for Consistency ---
    # 1. Verify the first state has exactly one entry
    assert len(collection_state_miss) == 1, \
        f"Expected 1 entry in 'miss' state, found {len(collection_state_miss)}"
    
    # 2. Verify the second state has exactly two entries
    assert len(collection_state_hit) == 2, \
        f"Expected 2 entries in 'hit' state, found {len(collection_state_hit)}"
    
    # 3. Verify the first entry is preserved in the second state (Consistency check)
    assert collection_state_hit[0] == collection_state_miss[0], \
        "The first queue runner entry was lost or modified in the second state"
        
    # 4. Verify the new entry is present
    assert collection_state_hit[1] == qr2, \
        "The second queue runner entry was not added correctly"

    print("Test Passed: Queue runner collection entries are consistent across states.")

if __name__ == "__main__":
    test_queue_runner_collection_consistency()