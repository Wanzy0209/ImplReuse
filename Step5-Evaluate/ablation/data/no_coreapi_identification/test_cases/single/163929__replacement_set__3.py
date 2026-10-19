import torch
import tensorflow as tf

def foo(x):
    x.tan_()
    x = x.t()
    # Replacement: x.argmin -> torch.backends.cudnn.version
    return torch.backends.cudnn.version()

# Replacement: torch.manual_seed -> tf.compat.v1.assign_sub
# We create a dummy variable to satisfy the signature of assign_sub
seed_var = tf.Variable(0)
tf.compat.v1.assign_sub(seed_var, 0)

# Replacement: torch.randn -> torch.backends.cudnn.version
x1 = torch.backends.cudnn.version()

# Replacement: x1.clone -> torch.backends.cudnn.version
x2 = torch.backends.cudnn.version()

out1 = foo(x1)

# Replacement: torch.compile -> torch.prod
cf = torch.prod(foo)
out2 = cf(x2)

# Replacement: torch.testing.assert_close -> torch.backends.cusparselt.version
torch.backends.cusparselt.version(out1, out2)