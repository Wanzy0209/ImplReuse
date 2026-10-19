```python
import tensorflow as tf

def cli_main():
    # Conversion: torch.set_float32_matmul_precision("high") enables TensorFloat-32 (TF32) on supported hardware.
    # TensorFlow equivalent is enabling TF32 execution via experimental config.
    # Note: TF32 is enabled by default on Ampere+ GPUs in TensorFlow.
    tf.config.experimental.enable_tensor_float_32_execution(True)

if __name__ == '__main__':
    cli_main()
```