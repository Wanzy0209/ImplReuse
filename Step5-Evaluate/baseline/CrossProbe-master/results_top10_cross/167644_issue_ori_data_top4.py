```python
import os
import tensorflow as tf

def cli_main():
    # PyTorch sets float32 matmul precision to "high" to enable TensorFloat-32 (TF32).
    # In TensorFlow, TF32 is controlled by the environment variable TF_ENABLE_TF32.
    # It is enabled by default on supported hardware (Ampere+), but we set it explicitly here.
    os.environ['TF_ENABLE_TF32'] = '1'

if __name__ == '__main__':
    cli_main()
```