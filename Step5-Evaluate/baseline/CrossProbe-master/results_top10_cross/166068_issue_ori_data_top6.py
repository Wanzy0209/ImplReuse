```python
import tensorflow as tf
from packaging import version

# TensorFlow uses sysconfig to access build details like ROCm version
print(version.parse(tf.sysconfig.get_build_info().get('rocm_version')))
```