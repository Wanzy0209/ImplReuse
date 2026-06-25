import torch

class TestMaxAutotune:
    def test_empty_conv_input_search_space_EXHAUSTIVE_kernel_size_3(self):
        # Test case that fails on XPU
        pass

if __name__ == '__main__':
    test = TestMaxAutotune()
    test.test_empty_conv_input_search_space_EXHAUSTIVE_kernel_size_3()