```python
import tensorflow as tf

def cli_main():
    # PyTorch's "high" precision enables TensorFloat-32 (TF32) on supported hardware.
    # TensorFlow equivalent: enable TF32 execution for matmul.
    tf.config.experimental.enable_tensor_float_32_execution(True)

if __name__ == '__main__':
    cli_main()
```