Using pip 25.3 from /usr/local/lib/python3.12/dist-packages/pip (python 3.12)
Obtaining file:///workspace/pytorch
  Running command Checking if build backend supports build_editable
  Checking if build backend supports build_editable ... done
  Running command Preparing editable metadata (pyproject.toml)
  Building wheel torch-2.10.0a0+git192034c
  Found cmake (/usr/local/bin/cmake) version: 3.31.6 (>=3.27)
  Traceback (most recent call last):
    File "/usr/local/lib/python3.12/dist-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 389, in <module>
      main()
    File "/usr/local/lib/python3.12/dist-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 373, in main
      json_out["return_val"] = hook(**hook_input["kwargs"])
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/usr/local/lib/python3.12/dist-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 209, in prepare_metadata_for_build_editable
      return hook(metadata_directory, config_settings)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/usr/local/lib/python3.12/dist-packages/setuptools/build_meta.py", line 478, in prepare_metadata_for_build_editable
      return self.prepare_metadata_for_build_wheel(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/usr/local/lib/python3.12/dist-packages/setuptools/build_meta.py", line 374, in prepare_metadata_for_build_wheel
      self.run_setup()
    File "/usr/local/lib/python3.12/dist-packages/setuptools/build_meta.py", line 317, in run_setup
      exec(code, locals())
    File "<string>", line 1784, in <module>
    File "<string>", line 1650, in main
    File "<string>", line 659, in mirror_inductor_external_kernels
  RuntimeError: Check the file paths in `mirror_inductor_external_kernels()`