```python
import tensorflow as tf

def cli_main():
    # Conversion: torch.set_float32_matmul_precision("high") enables TF32 on Ampere GPUs.
    # In TensorFlow, this is controlled via the experimental config.
    tf.config.experimental.enable_tensor_float_32_execution(True)

if __name__ == '__main__':
    cli_main()
```