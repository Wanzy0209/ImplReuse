auto m = py::handle(module).cast<py::module>();
-  // hipUUID is defined in either cuda.h or hip/driver_types.h
+  // hipUUID is defined in either cuda.h or hip/hip/driver_types.h
   // hipified to hipUUID which is defined in hip_runtime_api.h