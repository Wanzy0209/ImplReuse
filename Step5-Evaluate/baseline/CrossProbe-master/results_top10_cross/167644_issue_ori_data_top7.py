```python
import os
import tensorflow as tf

def cli_main():
    # PyTorch's set_float32_matmul_precision("high") enables TensorFloat-32 (TF32) for matmuls.
    # TensorFlow uses TF32 by default on Ampere GPUs.
    # To explicitly control TF32, the environment variable TF_ENABLE_TF32 is used.
    # Note: This must be set before TensorFlow initializes the GPU.
    os.environ['TF_ENABLE_TF32'] = '1'

if __name__ == '__main__':
    cli_main()
```