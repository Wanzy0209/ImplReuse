import tensorflow as tf
from tensorflow.python.framework import test_util

@test_util.run_all_in_graph_and_eager_modes
class TestGPUBuildSupport(tf.test.TestCase):
  def test_is_built_with_gpu_support(self):
    """
    Test case to verify the build configuration regarding GPU support.
    
    This test reflects the logic of the reported issue (PyTorch CUDA13 Binary 
    Cannot Be Built with SM_75 with NVSHMEM), where specific build configurations
    and symbol availability (linking) were problematic. 
    
    Here, we use the similar API `tf.test.is_built_with_gpu_support` to 
    verify that the build flags correctly reflect the availability of 
    CUDA/GPU capabilities, ensuring that tests relying on these features 
    are skipped or executed based on the actual build environment.
    """
    # Check the build-time flag for GPU support
    has_gpu_support = tf.test.is_built_with_gpu_support()
    
    # The original bug exposed undefined references (missing symbols) during linking.
    # This test ensures the API returns a valid boolean indicating the build status.
    self.assertIsInstance(has_gpu_support, bool)

    if has_gpu_support:
      # If the build claims GPU support, verify CUDA support is also present.
      # This aligns with the bug report's focus on CUDA 13 compatibility.
      self.assertTrue(tf.test.is_built_with_cuda(),
                      "is_built_with_gpu_support returned True, but is_built_with_cuda returned False.")
      
      # Verify that we can query physical devices without crashing.
      # A failure here (e.g., a segfault or symbol error) would be analogous 
      # to the nvlink errors in the original bug.
      gpu_devices = tf.config.list_physical_devices('GPU')
      # We don't assert len(gpu_devices) > 0 because the build might support GPU
      # even if no hardware is present on the test runner.
    else:
      # If no GPU support is claimed, ensure CUDA is also not supported.
      self.assertFalse(tf.test.is_built_with_cuda(),
                       "is_built_with_gpu_support returned False, but is_built_with_cuda returned True.")

if __name__ == '__main__':
  tf.test.main()