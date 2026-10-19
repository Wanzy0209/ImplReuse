```python
import tensorflow as tf
from packaging import version

# Conversion: torch.version.hip is replaced by accessing the ROCm version from sysconfig
print(version.parse(tf.sysconfig.get_build_info().get('rocm_version')))
```