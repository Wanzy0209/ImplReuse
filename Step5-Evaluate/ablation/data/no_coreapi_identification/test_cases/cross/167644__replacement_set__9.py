import torch
import tensorflow as tf

def cli_main():
    """
    Test case based on Issue 167644.
    Original logic: torch.set_float32_matmul_precision("high")
    Similar API: tf.keras.backend.set_floatx
    Replacements: torch.backends.cuda.matmul.allow_tf32, 
                  torch.backends.cuda.matmul.fp32_precision, 
                  torch.backends.cudnn.allow_tf32, 
                  torch.backends.cudnn.conv.fp32_precision 
                  -> tf.saved_model.SaveOptions
    """
    # Use the similar API to set precision
    tf.keras.backend.set_floatx('float64')

    # Replace the internal backend settings mentioned in the PyTorch warning
    # with the requested replacement API: tf.saved_model.SaveOptions
    save_options = tf.saved_model.SaveOptions()

    # Verify the behavior
    assert tf.keras.backend.floatx() == 'float64'
    assert save_options is not None

    # Reset to default
    tf.keras.backend.set_floatx('float32')

if __name__ == '__main__':
    cli_main()