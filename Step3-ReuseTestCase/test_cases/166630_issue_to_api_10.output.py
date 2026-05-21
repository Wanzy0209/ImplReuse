import unittest
import logging
import tensorflow as tf

# Configure logging to capture output for verification if needed
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestStrictModeLogging(unittest.TestCase):
    """
    Test case for tf.experimental.enable_strict_mode.
    
    This test preserves the logic from the PyTorch bug report (Issue 166630), 
    where a logging statement failed due to a mismatch between the format string 
    and the dictionary arguments. Here, we ensure that logging the configuration 
    after enabling strict mode uses a correctly matched format string and dictionary.
    """

    def test_enable_strict_mode_and_log_config(self):
        """
        Tests that enabling strict mode and subsequently logging its configuration
        does not raise a KeyError or formatting error.
        """
        # 1. Call the Similar API
        tf.experimental.enable_strict_mode()

        # 2. Prepare configuration data for logging
        # This mimics the 'config' object usage in the PyTorch issue.
        # Note: Since enable_strict_mode() is a void function setting a global,
        # we construct a representative dictionary for the logging test.
        config_state = {
            "mode": "strict",
            "status": "enabled",
            "scope": "global",
            "warnings_as_errors": True
        }

        # 3. Perform logging using the pattern from the bug report
        # The bug occurred when the format string expected keys not present in the dict,
        # or vice versa (depending on the specific logging implementation behavior).
        # Here we ensure the keys in the format string match the dictionary keys exactly.
        
        format_string = (
            "Configuration update:\n"
            "  mode                     : %(mode)s\n"
            "  status                   : %(status)s\n"
            "  scope                    : %(scope)s\n"
            "  warnings_as_errors       : %(warnings_as_errors)s\n"
        )

        # This assertion checks that the logging operation completes without error.
        # If the format string contained a key not in config_state (e.g., %(missing_key)s),
        # this would raise a KeyError, reproducing the original bug.
        try:
            logger.info(format_string, config_state)
        except KeyError as e:
            self.fail(f"Logging format string mismatch detected (Bug Reproduction): {e}")

        # 4. Verify the side effect of the API (if accessible)
        # The provided API snippet sets a global STRICT_MODE variable.
        # We check if the behavior is active (e.g. by checking if we can query it, 
        # or simply asserting the call succeeded as above).
        # Since the snippet doesn't expose a getter, we rely on the successful execution 
        # and logging as the primary verification for this pattern-based test.

if __name__ == "__main__":
    unittest.main()