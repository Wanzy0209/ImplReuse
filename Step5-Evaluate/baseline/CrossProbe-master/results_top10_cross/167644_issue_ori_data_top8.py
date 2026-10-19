```python
import tensorflow as tf

def cli_main():
    # PyTorch: torch.set_float32_matmul_precision("high")
    # TensorFlow: Enable TensorFloat-32 (TF32) execution on supported hardware (e.g., Ampere GPUs).
    # This allows faster matrix multiplication with slightly reduced precision, similar to PyTorch's "high" setting.
    tf.config.experimental.enable_tensor_float_32_execution(True)

if __name__ == '__main__':
    cli_main()
```