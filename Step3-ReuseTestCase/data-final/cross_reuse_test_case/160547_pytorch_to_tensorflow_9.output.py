import tensorflow as tf
from collections import namedtuple

def test_reparameterization_type_with_namedtuple():
    """
    Adapted test case for tf.compat.v1.distributions.ReparameterizationType.
    The original bug involved NamedTuple inputs failing in a non-strict export context.
    Here, we verify that the TensorFlow API correctly identifies the reparameterization
    type of distributions initialized with NamedTuple inputs.
    """
    # Setup: Define a NamedTuple similar to the original bug report
    DistParams = namedtuple('DistParams', 'loc scale')
    inp = DistParams(0.0, 1.0)

    # Context: Create a distribution using the NamedTuple input
    # Normal distribution is expected to be FULLY_REPARAMETERIZED
    normal_dist = tf.compat.v1.distributions.Normal(loc=inp.loc, scale=inp.scale)
    
    # Verification 1: Check the reparameterization type attribute
    print(f"Normal Distribution Reparameterization Type: {normal_dist.reparameterization_type}")
    assert normal_dist.reparameterization_type == tf.compat.v1.distributions.FULLY_REPARAMETERIZED

    # Verification 2: Contrast with a non-reparameterized distribution
    # Empirical distribution is expected to be NOT_REPARAMETERIZED
    emp_dist = tf.compat.v1.distributions.Empirical(params=[0.0, 1.0])
    print(f"Empirical Distribution Reparameterization Type: {emp_dist.reparameterization_type}")
    assert emp_dist.reparameterization_type == tf.compat.v1.distributions.NOT_REPARAMETERIZED

    print("Test passed: Reparameterization types are correctly identified.")

if __name__ == "__main__":
    test_reparameterization_type_with_namedtuple()