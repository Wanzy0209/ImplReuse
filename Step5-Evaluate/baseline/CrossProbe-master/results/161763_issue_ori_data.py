```python
import tensorflow as tf


def foo(x):
    # Conversion: torch.tensor -> tf.constant
    c = tf.constant(7, dtype=tf.uint8)
    # Conversion: torch.neg -> tf.negative
    return c + x, tf.negative(c), tf.negative(c) + x
    

# Conversion: torch.manual_seed -> tf.random.set_seed
tf.random.set_seed(0)
# Conversion: torch.randn -> tf.random.normal
x = tf.random.normal((2, 2), dtype=tf.float32)
print(f"input: {x}")
'''
input: tensor([[ 1.5410, -0.2934],
        [-2.1788,  0.5684]])
'''
# Conversion: torch.compile -> tf.function
cfoo = tf.function(foo)
res = foo(x)
cres = cfoo(x)

print(f"res[0]: {res[0]}")
print(f"cres[0]: {cres[0]}")
'''
res[0]: tensor([[8.5410, 6.7066],
        [4.8212,  7.5684]])
cres[0]: tensor([[8.5410, 6.7066],
        [4.8212,  7.5684]])
'''
print(f"res[1]: {res[1]}")
print(f"cres[1]: {cres[1]}")
'''
res[1]: 249
cres[1]: 249
'''
print(f"res[2]: {res[2]}")
print(f"cres[2]: {cres[2]}")
'''
res[2]: tensor([[250.5410, 248.7066],
        [246.8212, 249.5684]])
cres[2]: tensor([[-5.4590, -7.2934],
        [-9.1788, -6.4316]])
'''
```