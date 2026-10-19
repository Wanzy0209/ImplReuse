```python
import tensorflow as tf

def cli_main():
    # PyTorch's "high" precision allows TensorFloat-32 (TF32) on supported hardware.
    # TensorFlow equivalent is enabling TF32 execution.
    tf.config.experimental.enable_tensor_float_32_execution(True)

if __name__ == '__main__':
    cli_main()
```