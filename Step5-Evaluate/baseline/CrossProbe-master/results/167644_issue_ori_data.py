```python
import tensorflow as tf

def cli_main():
    # Conversion: torch.set_float32_matmul_precision("high")
    # PyTorch "high" precision enables TensorFloat-32 (TF32) on Ampere+ GPUs.
    # TensorFlow equivalent is enabling TF32 execution via experimental config.
    tf.config.experimental.enable_tensor_float_32_execution(True)

if __name__ == '__main__':
    cli_main()
```