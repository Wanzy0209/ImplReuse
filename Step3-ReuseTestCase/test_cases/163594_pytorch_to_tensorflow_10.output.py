import unittest
import subprocess
import sys
import tempfile
import os

class TestRangeInputProducerTimeout(unittest.TestCase):
    """
    Adapted from PyTorch Issue 163594: test_dtensor_compile_redistribute.
    
    The original test failed due to a subprocess.TimeoutExpired, indicating a hang
    or deadlock during the execution of a compiled graph with distributed tensors.
    
    This test adapts the logic to the TensorFlow API `tf.compat.v1.train.range_input_producer`.
    It verifies that the TensorFlow input pipeline (which involves queue runners and
    session management) executes successfully within a reasonable timeout, preventing
    similar hang/deadlock scenarios.
    """

    def _get_payload_script(self):
        """
        Generates a Python script that uses tf.compat.v1.train.range_input_producer
        to simulate a data redistribution/processing pipeline.
        """
        return """
import tensorflow as tf
import sys

# Disable eager execution to use v1 queues and sessions
tf.compat.v1.disable_eager_execution()

def main():
    # Configuration mimicking a distributed data workload
    limit = 100
    num_epochs = 2
    capacity = 32
    
    with tf.compat.v1.Session() as sess:
        # Create the range input producer (similar to distributing data)
        # Using shuffle=True to add complexity similar to redistribution
        producer = tf.compat.v1.train.range_input_producer(
            limit=limit,
            num_epochs=num_epochs,
            shuffle=True,
            seed=42,
            capacity=capacity
        )
        
        # Dequeue operation to consume the data
        dequeue_op = producer.dequeue()
        
        # Initialize local variables (required for num_epochs)
        sess.run([tf.compat.v1.local_variables_initializer()])
        
        # Start queue runners (essential for the producer to work)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        try:
            # Process the data
            # This loop should complete deterministically based on limit and num_epochs
            for _ in range(limit * num_epochs):
                res = sess.run(dequeue_op)
                # Simulate minimal processing
                pass 
        except tf.errors.OutOfRangeError:
            # Expected when num_epochs are consumed
            pass
        except Exception as e:
            print(f"Error during execution: {e}", file=sys.stderr)
            sys.exit(1)
        finally:
            # Ensure clean shutdown to prevent hangs
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    main()
"""

    def test_range_input_producer_timeout(self):
        """
        Executes the payload script in a subprocess with a timeout.
        Reproduces the testing methodology of the original PyTorch bug report
        to catch potential deadlocks or infinite loops in the TensorFlow API.
        """
        script_content = self._get_payload_script()
        
        # Create a temporary file for the payload script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            f.flush()
            temp_path = f.name

        try:
            # Run the script with a timeout (30 seconds, matching the original bug report)
            res = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Check that the process exited successfully
            self.assertEqual(
                res.returncode, 
                0, 
                f"Script failed with return code {res.returncode}.\nStdout: {res.stdout}\nStderr: {res.stderr}"
            )
            
        except subprocess.TimeoutExpired as e:
            # This reproduces the original bug condition
            self.fail(
                f"Test timed out after 30 seconds, reproducing the hang behavior.\n"
                f"Stdout: {e.stdout}\nStderr: {e.stderr}"
            )
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == '__main__':
    unittest.main()