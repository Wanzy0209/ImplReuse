#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12090)
#define C10_NVML_DRIVER_API_OPTIONAL(_) _(nvmlDeviceGetGpuFabricInfoV)
#endif