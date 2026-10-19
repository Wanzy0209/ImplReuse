import torch
import tensorflow as tf
import warnings

def cli_main():
    """
    Test case adapted from PyTorch issue #167644.
    Original issue: Calling torch.set_float32_matmul_precision triggers warnings about internal settings.
    Similar API: tf.keras.backend.set_floatx.
    Replacements: Internal settings (allow_tf32, etc.) are mapped to tf.compat.v1.disable_v2_behavior.
    """
    
    # Original logic: torch.set_float32_matmul_precision("high")
    # Translated to Similar API: tf.keras.backend.set_floatx
    # We set it to 'float32' to match the standard precision setting context.
    tf.keras.backend.set_floatx('float32')

    # Replacements for the internal settings mentioned in the bug report's warning:
    # - torch.backends.cuda.matmul.allow_tf32
    # - torch.backends.cuda.matmul.fp32_precision
    # - torch.backends.cudnn.allow_tf32
    # - torch.backends.cudnn.conv.fp32_precision
    # All are replaced by: tf.compat.v1.disable_v2_behavior
    # This call represents the interaction with the "internal" or "legacy" behavior flags.
    tf.compat.v1.disable_v2_behavior()

    # Assertion to verify the main API call worked as expected
    assert tf.keras.backend.floatx() == 'float32', "Floatx was not set correctly"

if __name__ == '__main__':
    # Suppress TF info logs to keep output minimal, focusing on potential warnings
    tf.get_logger().setLevel('ERROR')
    cli_main()