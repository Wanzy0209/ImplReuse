In file included from /home/coder/pytorch/c10/cuda/CUDACachingAllocator.cpp:19:
  /home/coder/pytorch/c10/cuda/driver_api.h:70:43: error: 'nvmlDeviceGetGpuFabricInfoV' was not declared in this scope; did you mean 'nvmlDeviceGetGpuFabricInfo'?
     70 | #define C10_NVML_DRIVER_API_OPTIONAL(_) _(nvmlDeviceGetGpuFabricInfoV)
        |                                           ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  /home/coder/pytorch/c10/cuda/driver_api.h:76:39: note: in definition of macro 'CREATE_MEMBER'
     76 | #define CREATE_MEMBER(name) decltype(&name) name##_;
        |                                       ^~~~
  /home/coder/pytorch/c10/cuda/driver_api.h:80:3: note: in expansion of macro 'C10_NVML_DRIVER_API_OPTIONAL'
     80 |   C10_NVML_DRIVER_API_OPTIONAL(CREATE_MEMBER)