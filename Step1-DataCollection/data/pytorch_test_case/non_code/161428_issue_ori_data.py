python setup.py develop
Building wheel torch-2.8.0a0+gitba56102
-- Building version 2.8.0a0+gitba56102
-- Checkout nccl release tag: v2.27.3-1
cmake --build . --target install --config Release
ninja: error: rebuilding 'build.ninja': '/home/yinghai/pytorch/torchgen/executorch/__init__.py', needed by 'aten/src/ATen/core_generated_declarations_yaml.cmake', missing and no known rule to make it