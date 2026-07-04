import unittest
import logging
import io
import torch

class TestCPUCapabilityLogging(unittest.TestCase):
    """
    Test case to verify logging behavior with torch.backends.cpu.get_cpu_capability.
    This test is based on the bug pattern found in torch.distributed.launcher.api.py
    where a mismatch between the format string and the dictionary arguments caused
    a logging error.
    """
    
    def setUp(self):
        # Configure a logger to capture output for verification
        self.logger = logging.getLogger("test_cpu_logger")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicate logs in test environment
        self.logger.handlers = []
        
        # Create a string stream to capture log output
        self.log_stream = io.StringIO()
        self.handler = logging.StreamHandler(self.log_stream)
        self.handler.setLevel(logging.INFO)
        self.logger.addHandler(self.handler)

    def test_logging_format_string_consistency(self):
        """
        Tests that logging the CPU capability using a format string and a dictionary
        works correctly, ensuring no KeyError is raised due to missing arguments
        (similar to the 'signals_to_handle' issue in the bug report).
        """
        # Get the CPU capability using the similar API
        cpu_capability = torch.backends.cpu.get_cpu_capability()
        
        # Define the format string. Note: The key 'cpu_capability' must exist
        # in the dictionary passed to logger.info to avoid the bug.
        format_string = (
            "System Configuration:\n"
            "  cpu_capability          : %(cpu_capability)s\n"
        )
        
        data_dict = {
            "cpu_capability": cpu_capability
        }

        # Execute the logging call. 
        # In the original bug, this would raise a KeyError if the format string
        # expected a key not present in the dict, or vice-versa depending on implementation.
        try:
            self.logger.info(format_string, data_dict)
        except KeyError as e:
            self.fail(f"Logging raised a KeyError due to format string mismatch: {e}")
        except Exception as e:
            self.fail(f"Logging raised an unexpected exception: {e}")

        # Verify that the output contains the expected information
        log_contents = self.log_stream.getvalue()
        self.assertIn("System Configuration", log_contents)
        self.assertIn(cpu_capability, log_contents)
        self.assertIn("cpu_capability", log_contents)

if __name__ == '__main__':
    unittest.main()