File "/home/usr/pytorchtorch/_inductor/analysis/profile_analysis.py", line 801, in main
    p.augment_trace()
  File "/home/usr/pytorchtorch/_inductor/analysis/profile_analysis.py", line 494, in augment_trace
    self.data = _augment_trace_helper(self.data)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/usr/pytorchtorch/_inductor/analysis/profile_analysis.py", line 348, in _augment_trace_helper
    flops = _calculate_flops(external_op)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/usr/pytorchtorch/_inductor/analysis/profile_analysis.py", line 205, in _calculate_flops
    return flop_function(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/usr/pytorchtorch/utils/flop_counter.py", line 33, in nf
    return f(*args, out_shape=out_shape, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/usr/pytorchtorch/utils/flop_counter.py", line 158, in conv_flop
    return conv_flop_count(x_shape, w_shape, out_shape, transposed=transposed)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/usr/pytorchtorch/utils/flop_counter.py", line 132, in conv_flop_count
    conv_shape = (x_shape if transposed else out_shape)[2:]
                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^
TypeError: 'NoneType' object is not subscriptable

To execute this test, run the following from the base repo dir:
    python test/inductor/test_analysis.py TestAnalysisXPU.test_augment_trace_against_flop_counter_maxat0_xpu_float16