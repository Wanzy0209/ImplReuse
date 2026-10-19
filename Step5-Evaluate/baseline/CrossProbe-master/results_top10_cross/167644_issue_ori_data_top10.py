```python
import tensorflow as tf

def cli_main():
    # Conversion: torch.set_float32_matmul_precision("high") enables TensorFloat-32 (TF32)
    # on supported hardware (e.g., NVIDIA Ampere+). In TensorFlow, this is controlled
    # via the experimental configuration.
    tf.config.experimental.enable_tensor_float_32_execution(True)

if __name__ == '__main__':
    cli_main()
```