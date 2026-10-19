```python
import tensorflow as tf
from packaging import version

# torch.version.hip provides the HIP version; in TensorFlow, retrieve the ROCm version from build info
print(version.parse(tf.sysconfig.get_build_info().get('rocm_version')))
```