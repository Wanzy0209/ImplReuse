Traceback (most recent call last):
  File "/Users/ec2-user/runner/_work/pytorch/pytorch/test/quantization/fx/test_model_report_fx.py", line 348, in test_conv_sub_class_considered
    with override_quantized_engine('qnnpack'):
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Cellar/python@3.12/3.12.11/Frameworks/Python.framework/Versions/3.12/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/Users/ec2-user/runner/_work/_temp/venv-3.12-1758080236/lib/python3.12/site-packages/torch/testing/_internal/common_quantized.py", line 151, in override_quantized_engine
    torch.backends.quantized.engine = previous
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/ec2-user/runner/_work/_temp/venv-3.12-1758080236/lib/python3.12/site-packages/torch/backends/quantized/__init__.py", line 37, in __set__
    torch._C._set_qengine(_get_qengine_id(val))
RuntimeError: quantized engine NoQEngine is not supported

To execute this test, run the following from the base repo dir:
    python test/quantization/fx/test_model_report_fx.py TestFxModelReportDetector.test_conv_sub_class_considered

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0