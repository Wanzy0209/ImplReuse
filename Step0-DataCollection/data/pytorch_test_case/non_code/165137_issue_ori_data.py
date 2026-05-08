Caught exception: 
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/torch/testing/_internal/common_distributed.py", line 864, in run_test
    getattr(self, test_name)()
  File "/usr/local/lib/python3.12/dist-packages/torch/testing/_internal/common_distributed.py", line 712, in wrapper
    fn()
  File "/usr/local/lib/python3.12/dist-packages/torch/testing/_internal/common_utils.py", line 3277, in wrapper
    method(*args, **kwargs)
  File "/usr/local/lib/python3.12/dist-packages/torch/testing/_internal/distributed/_tensor/common_dtensor.py", line 509, in wrapper
    raise e
  File "/usr/local/lib/python3.12/dist-packages/torch/testing/_internal/distributed/_tensor/common_dtensor.py", line 506, in wrapper
    func(self, *args, **kwargs)  # type: ignore[misc]
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/testing/_internal/common_distributed.py", line 219, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/testing/_internal/distributed/checkpoint_utils.py", line 154, in wrapper
    func(self, *args, **kwargs)
  File "/opt/pytorch/pytorch/test/distributed/checkpoint/e2e/test_fsdp_ep.py", line 76, in test_e2e
    mesh_fsdp_ep = _mesh_resources.create_sub_mesh(mesh_fsdp_tp, ("dp",), [(0,)])
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/distributed/device_mesh.py", line 104, in create_sub_mesh
    flatten_mesh = self.root_to_flatten_mapping[device_mesh][name]
                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
KeyError: "DeviceMesh((dp=2, tp=4), device: 'cuda', stride: (4, 1))"
To execute this test, run the following from the base repo dir:
    python test/distributed/checkpoint/e2e/test_fsdp_ep.py TestFSDPWithEP.test_e2e