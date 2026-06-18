import tensorflow as tf
import unittest
import tempfile
import shutil
import logging

class TestProfilerStartContext(unittest.TestCase):
    def test_profiler_start_duplicate_warning(self):
        """
        Test that tf.profiler.experimental.start logs a warning 
        if called multiple times, reflecting the logging pattern 
        seen in the similar API (heartbeat.start).
        """
        log_dir = tempfile.mkdtemp()
        
        try:
            # Capture logs to verify the context/warning is provided
            with self.assertLogs(level='WARNING') as cm:
                # First start
                tf.profiler.experimental.start(log_dir)
                
                # Second start - based on the similar API pattern, 
                # this should trigger a warning about an existing session.
                tf.profiler.experimental.start(log_dir)
                
                # Verify that a warning was logged, indicating context preservation
                self.assertTrue(len(cm.output) > 0)
                self.assertTrue(any("already" in message.lower() or "running" in message.lower() 
                                    for message in cm.output))
            
            # Stop the profiler to clean up state
            tf.profiler.experimental.stop()
            
        finally:
            # Clean up the temporary directory
            if tf.io.gfile.exists(log_dir):
                tf.io.gfile.rmtree(log_dir)

if __name__ == "__main__":
    unittest.main()