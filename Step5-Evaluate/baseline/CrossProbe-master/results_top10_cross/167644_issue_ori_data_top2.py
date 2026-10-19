```python
import tensorflow as tf

def cli_main():
    # PyTorch: torch.set_float32_matmul_precision("high")
    # TensorFlow equivalent: Enable TensorFloat-32 (TF32) execution on supported hardware (Ampere+ GPUs)
    tf.config.experimental.enable_tensor_float_32_execution(True)

if __name__ == '__main__':
    cli_main()
```