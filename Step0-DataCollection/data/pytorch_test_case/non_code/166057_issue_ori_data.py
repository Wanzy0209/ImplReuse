-- ******** Summary ********
-- General:
--   CMake version         : 4.0.0
--   CMake command         : /usr/local/lib/python3.12/dist-packages/cmake/data/bin/cmake
--   System                : Linux
--   C++ compiler          : /usr/bin/riscv64-linux-gnu-g++-14
--   C++ compiler id       : GNU
--   C++ compiler version  : 14.2.0
--   Using ccache if found : ON
--   Found ccache          : /usr/bin/ccache
--   CXX flags             : -march=rv64gcv -fcf-protection=none -fvisibility-inlines-hidden -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DLIBKINETO_NOXPUPTI=ON -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -DC10_NODEPRECATED -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=range-loop-construct -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-unused-parameter -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wsuggest-override -Wno-psabi -Wno-error=old-style-cast -faligned-new -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-dangling-reference -Wno-error=dangling-reference -Wno-stringop-overflow
--   Shared LD flags       :  -Wl,--no-as-needed -rdynamic
--   Static LD flags       : 
--   Module LD flags       : 
--   Build type            : Release
--   Compile definitions   : ONNX_ML=1;ONNXIFI_ENABLE_EXT=1;ONNX_NAMESPACE=onnx_torch;HAVE_MMAP=1;_FILE_OFFSET_BITS=64;HAVE_SHM_OPEN=1;HAVE_SHM_UNLINK=1;HAVE_MALLOC_USABLE_SIZE=1;HAVE_POSIX_FALLOCATE=1;USE_EXTERNAL_MZCRC;MINIZ_DISABLE_ZIP_READER_CRC32_CHECKS
--   CMAKE_PREFIX_PATH     : 
--   CMAKE_INSTALL_PREFIX  : /usr/local
--   USE_GOLD_LINKER       : OFF
-- 
--   TORCH_VERSION         : 2.10.0
--   BUILD_STATIC_RUNTIME_BENCHMARK: OFF
--   BUILD_BINARY          : OFF
--   BUILD_CUSTOM_PROTOBUF : ON
--     Link local protobuf : ON
--   BUILD_PYTHON          : ON
--     Python version      : 3.12.3
--     Python executable   : /workspace/venv/bin/python3
--     Python library      : 
--     Python includes     : /opt/sysroot/include/python3.12
--     Python site-package : /workspace/venv/lib/python3.12/site-packages
--   BUILD_SHARED_LIBS     : ON
--   CAFFE2_USE_MSVC_STATIC_RUNTIME     : OFF
--   BUILD_TEST            : OFF
--   BUILD_JNI             : OFF
--   BUILD_MOBILE_AUTOGRAD : OFF
--   BUILD_LITE_INTERPRETER: OFF
--   INTERN_BUILD_MOBILE   : 
--   TRACING_BASED         : OFF
--   USE_BLAS              : 0
--   USE_LAPACK            : 0
--   USE_ASAN              : OFF
--   USE_LSAN              : OFF
--   USE_TSAN              : OFF
--   USE_CPP_CODE_COVERAGE : OFF
--   USE_CUDA              : OFF
--   USE_XPU               : OFF
--   USE_ROCM              : OFF
--   BUILD_NVFUSER         : 
--   USE_EIGEN_FOR_BLAS    : ON
--   USE_EIGEN_FOR_SPARSE  : OFF
--   USE_FBGEMM            : OFF
--   USE_FBGEMM_GENAI      : OFF
--   USE_KINETO            : ON
--   USE_GFLAGS            : OFF
--   USE_GLOG              : OFF
--   USE_LITE_PROTO        : OFF
--   USE_PYTORCH_METAL     : OFF
--   USE_PYTORCH_METAL_EXPORT     : OFF
--   USE_MPS               : OFF
--   CAN_COMPILE_METAL     : 
--   USE_MKL               : OFF
--   USE_MKLDNN            : OFF
--   USE_PRIORITIZED_TEXT_FOR_LD : OFF
--   USE_UCC               : OFF
--   USE_ITT               : OFF
--   USE_XCCL              : OFF
--   USE_NCCL              : OFF
--   Found NVSHMEM         : 
--   USE_NNPACK            : OFF
--   USE_NUMPY             : OFF
--   USE_OBSERVERS         : ON
--   USE_OPENCL            : OFF
--   USE_OPENMP            : ON
--   USE_MIMALLOC          : OFF
--   USE_VULKAN            : OFF
--   USE_PROF              : OFF
--   USE_PYTORCH_QNNPACK   : OFF
--   USE_XNNPACK           : OFF
--   USE_DISTRIBUTED       : ON
--     USE_MPI               : OFF
--     USE_GLOO              : ON
--     USE_GLOO_WITH_OPENSSL : OFF
--     USE_GLOO_IBVERBS      : OFF
--     USE_TENSORPIPE        : ON
--   Public Dependencies  : 
--   Private Dependencies : Threads::Threads;cpuinfo;fp16;caffe2::openmp;tensorpipe;nlohmann;moodycamel;gloo;rt;fmt::fmt-header-only;kineto;gcc_s;gcc;dl
--   Public CUDA Deps.    : 
--   Private CUDA Deps.   : 
--   USE_COREML_DELEGATE     : OFF
--   BUILD_LAZY_TS_BACKEND   : ON
--   USE_ROCM_KERNEL_ASSERT : OFF
-- Configuring done (14.4s)
-- Generating done (0.5s)
-- Build files have been written to: /workspace/pytorch/build_docker
jenkins@localhost:/workspace/pytorch/build_docker$ make -j
...
[ 90%] Building CXX object caffe2/CMakeFiles/torch_cpu.dir/__/torch/csrc/api/src/serialize/output-archive.cpp.o
In file included from /usr/riscv64-linux-gnu/include/c++/14/riscv64-linux-gnu/bits/c++allocator.h:33,
                 from /usr/riscv64-linux-gnu/include/c++/14/bits/allocator.h:46,
                 from /usr/riscv64-linux-gnu/include/c++/14/memory:65,
                 from /workspace/pytorch/c10/util/Backtrace.h:5,
                 from /workspace/pytorch/c10/util/Exception.h:6,
                 from /workspace/pytorch/aten/src/ATen/BlasBackend.h:3,
                 from /workspace/pytorch/aten/src/ATen/Context.h:3,
                 from /workspace/pytorch/aten/src/ATen/ATen.h:7,
                 from /workspace/pytorch/torch/csrc/api/include/torch/types.h:3,
                 from /workspace/pytorch/torch/csrc/distributed/rpc/message.h:3,
                 from /workspace/pytorch/torch/csrc/distributed/rpc/rpc_command_base.h:3,
                 from /workspace/pytorch/torch/csrc/distributed/rpc/python_call.h:3,
                 from /workspace/pytorch/torch/csrc/distributed/rpc/python_call.cpp:1:
In member function 'void std::__new_allocator<_Tp>::deallocate(_Tp*, size_type) [with _Tp = char]',
    inlined from 'static void std::allocator_traits<std::allocator<_Tp1> >::deallocate(allocator_type&, pointer, size_type) [with _Tp = char]' at /usr/riscv64-linux-gnu/include/c++/14/bits/alloc_traits.h:513:23,
    inlined from 'std::vector<_Tp, _Alloc>::_M_realloc_append(_Args&& ...)::_Guard::~_Guard() [with _Args = {char}; _Tp = char; _Alloc = std::allocator<char>]' at /usr/riscv64-linux-gnu/include/c++/14/bits/vector.tcc:616:18,
    inlined from 'void std::vector<_Tp, _Alloc>::_M_realloc_append(_Args&& ...) [with _Args = {char}; _Tp = char; _Alloc = std::allocator<char>]' at /usr/riscv64-linux-gnu/include/c++/14/bits/vector.tcc:688:7,
    inlined from 'std::vector<_Tp, _Alloc>::reference std::vector<_Tp, _Alloc>::emplace_back(_Args&& ...) [with _Args = {char}; _Tp = char; _Alloc = std::allocator<char>]' at /usr/riscv64-linux-gnu/include/c++/14/bits/vector.tcc:123:21,
    inlined from 'void std::vector<_Tp, _Alloc>::push_back(value_type&&) [with _Tp = char; _Alloc = std::allocator<char>]' at /usr/riscv64-linux-gnu/include/c++/14/bits/stl_vector.h:1301:21,
    inlined from 'virtual c10::intrusive_ptr<torch::distributed::rpc::Message> torch::distributed::rpc::PythonCall::toMessageImpl() &&' at /workspace/pytorch/torch/csrc/distributed/rpc/python_call.cpp:12:20:
/usr/riscv64-linux-gnu/include/c++/14/bits/new_allocator.h:172:33: warning: 'void operator delete(void*, std::size_t)' called on pointer '<unknown>' with nonzero offset [1, 9223372036854775807] [-Wfree-nonheap-object]
  172 |         _GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
      |                                 ^
In member function '_Tp* std::__new_allocator<_Tp>::allocate(size_type, const void*) [with _Tp = char]',
    inlined from 'static _Tp* std::allocator_traits<std::allocator<_Tp1> >::allocate(allocator_type&, size_type) [with _Tp = char]' at /usr/riscv64-linux-gnu/include/c++/14/bits/alloc_traits.h:478:28,
    inlined from 'std::_Vector_base<_Tp, _Alloc>::pointer std::_Vector_base<_Tp, _Alloc>::_M_allocate(std::size_t) [with _Tp = char; _Alloc = std::allocator<char>]' at /usr/riscv64-linux-gnu/include/c++/14/bits/stl_vector.h:380:33,
    inlined from 'void std::vector<_Tp, _Alloc>::reserve(size_type) [with _Tp = char; _Alloc = std::allocator<char>]' at /usr/riscv64-linux-gnu/include/c++/14/bits/vector.tcc:79:33,
    inlined from 'virtual c10::intrusive_ptr<torch::distributed::rpc::Message> torch::distributed::rpc::PythonCall::toMessageImpl() &&' at /workspace/pytorch/torch/csrc/distributed/rpc/python_call.cpp:11:18:
/usr/riscv64-linux-gnu/include/c++/14/bits/new_allocator.h:151:55: note: returned from 'void* operator new(std::size_t)'
  151 |         return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
      |                                                       ^
during GIMPLE pass: lower
In file included from /workspace/pytorch/build_docker/aten/src/ATen/native/cpu/DepthwiseConvKernel.cpp.DEFAULT.cpp:1:
/workspace/pytorch/aten/src/ATen/native/cpu/DepthwiseConvKernel.cpp: In function 'void at::native::{anonymous}::winograd_f2k3_kernel_transform__rvv(vfloat32m1_t, vfloat32m1_t, vfloat32m1_t, vfloat32m1x4_t*)':
/workspace/pytorch/aten/src/ATen/native/cpu/DepthwiseConvKernel.cpp:307:44: internal compiler error: in gsi_replace, at gimple-iterator.cc:438
  307 |   *transform = __riscv_vset_v_f32m1_f32m1x4(*transform, 3, g2);
      |                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~
0x296a4cd internal_error(char const*, ...)
        ???:0
0xa08258 fancy_abort(char const*, int, char const*)
        ???:0
0x160c129 riscv_gimple_fold_builtin(gimple_stmt_iterator*)
        ???:0
Please submit a full bug report, with preprocessed source (by using -freport-bug).
Please include the complete backtrace with any bug report.
See <file:///usr/share/doc/gcc-14/README.Bugs> for instructions.
make[2]: *** [caffe2/CMakeFiles/torch_cpu.dir/build.make:7428: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/DepthwiseConvKernel.cpp.DEFAULT.cpp.o] Error 1
make[1]: *** [CMakeFiles/Makefile2:2827: caffe2/CMakeFiles/torch_cpu.dir/all] Error 2
make: *** [Makefile:146: all] Error 2
jenkins@localhost:/workspace/pytorch/build_docker$