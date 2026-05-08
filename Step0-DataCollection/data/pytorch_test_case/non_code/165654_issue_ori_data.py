$ lldb -- python -c "import torch;torch.nn.ConvTranspose1d(2016, 1026, 1024, stride=256)(torch.rand(1, 2016, 224))"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'lldb'
(lldb) target create "python"
Current executable set to 'python' (aarch64).
(lldb) settings set -- target.run-args  "-c" "import torch;torch.nn.ConvTranspose1d(2016, 1026, 1024, stride=256)(torch.rand(1, 2016, 224))"
(lldb) r
Process 1623 launched: '/opt/conda/envs/py_3.10/bin/python' (aarch64)
warning: (aarch64) /opt/conda/envs/py_3.10/lib/python3.10/site-packages/numpy.libs/libgfortran-daac5196.so.5.0.0 No LZMA support found for reading .gnu_debugdata section
Process 1623 stopped
* thread #1, name = 'python', stop reason = signal SIGSEGV: invalid address (fault address: 0xfff88d79d0c0)
    frame #0: 0x0000ffffe9002400 libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const + 512
libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const:
->  0xffffe9002400 <+512>: str    q0, [x26, w0, sxtw]
    0xffffe9002404 <+516>: ldp    x1, x0, [x19, #0x8]
    0xffffe9002408 <+520>: ldr    w0, [x0]
    0xffffe900240c <+524>: ldr    w1, [x1]
  thread #17, name = 'python', stop reason = signal SIGSEGV: invalid address (fault address: 0xfff88d71f0c0)
    frame #0: 0x0000ffffe9002400 libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const + 512
libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const:
->  0xffffe9002400 <+512>: str    q0, [x26, w0, sxtw]
    0xffffe9002404 <+516>: ldp    x1, x0, [x19, #0x8]
    0xffffe9002408 <+520>: ldr    w0, [x0]
    0xffffe900240c <+524>: ldr    w1, [x1]
  thread #18, name = 'python', stop reason = signal SIGSEGV: invalid address (fault address: 0xfff88d6a10c0)
    frame #0: 0x0000ffffe9002400 libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const + 512
libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const:
->  0xffffe9002400 <+512>: str    q0, [x26, w0, sxtw]
    0xffffe9002404 <+516>: ldp    x1, x0, [x19, #0x8]
    0xffffe9002408 <+520>: ldr    w0, [x0]
    0xffffe900240c <+524>: ldr    w1, [x1]
  thread #19, name = 'python', stop reason = signal SIGSEGV: invalid address (fault address: 0xfff88d6230c0)
    frame #0: 0x0000ffffe9002400 libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const + 512
libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const:
->  0xffffe9002400 <+512>: str    q0, [x26, w0, sxtw]
    0xffffe9002404 <+516>: ldp    x1, x0, [x19, #0x8]
    0xffffe9002408 <+520>: ldr    w0, [x0]
    0xffffe900240c <+524>: ldr    w1, [x1]
  thread #20, name = 'python', stop reason = signal SIGSEGV: invalid address (fault address: 0xfff88d5a50c0)
    frame #0: 0x0000ffffe9002400 libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const + 512
libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const:
->  0xffffe9002400 <+512>: str    q0, [x26, w0, sxtw]
    0xffffe9002404 <+516>: ldp    x1, x0, [x19, #0x8]
    0xffffe9002408 <+520>: ldr    w0, [x0]
    0xffffe900240c <+524>: ldr    w1, [x1]
  thread #21, name = 'python', stop reason = signal SIGSEGV: invalid address (fault address: 0xfff88d5270c0)
    frame #0: 0x0000ffffe9002400 libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const + 512
libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const:
->  0xffffe9002400 <+512>: str    q0, [x26, w0, sxtw]
    0xffffe9002404 <+516>: ldp    x1, x0, [x19, #0x8]
    0xffffe9002408 <+520>: ldr    w0, [x0]
    0xffffe900240c <+524>: ldr    w1, [x1]
  thread #22, name = 'python', stop reason = signal SIGSEGV: invalid address (fault address: 0xfff88d4a90c0)
    frame #0: 0x0000ffffe9002400 libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const + 512
libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const:
->  0xffffe9002400 <+512>: str    q0, [x26, w0, sxtw]
    0xffffe9002404 <+516>: ldp    x1, x0, [x19, #0x8]
    0xffffe9002408 <+520>: ldr    w0, [x0]
    0xffffe900240c <+524>: ldr    w1, [x1]
  thread #23, name = 'python', stop reason = signal SIGSEGV: invalid address (fault address: 0xfff88d42b0c0)
    frame #0: 0x0000ffffe9002400 libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const + 512
libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const:
->  0xffffe9002400 <+512>: str    q0, [x26, w0, sxtw]
    0xffffe9002404 <+516>: ldp    x1, x0, [x19, #0x8]
    0xffffe9002408 <+520>: ldr    w0, [x0]
    0xffffe900240c <+524>: ldr    w1, [x1]
  thread #24, name = 'python', stop reason = signal SIGSEGV: invalid address (fault address: 0xfff88d3ad0c0)
    frame #0: 0x0000ffffe9002400 libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const + 512
libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const:
->  0xffffe9002400 <+512>: str    q0, [x26, w0, sxtw]
    0xffffe9002404 <+516>: ldp    x1, x0, [x19, #0x8]
    0xffffe9002408 <+520>: ldr    w0, [x0]
    0xffffe900240c <+524>: ldr    w1, [x1]
  thread #25, name = 'python', stop reason = signal SIGSEGV: invalid address (fault address: 0xfff88d32f0c0)
    frame #0: 0x0000ffffe9002400 libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const + 512
libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const:
->  0xffffe9002400 <+512>: str    q0, [x26, w0, sxtw]
    0xffffe9002404 <+516>: ldp    x1, x0, [x19, #0x8]
    0xffffe9002408 <+520>: ldr    w0, [x0]
    0xffffe900240c <+524>: ldr    w1, [x1]
  thread #26, name = 'python', stop reason = signal SIGSEGV: invalid address (fault address: 0xfff88d2b10c0)
    frame #0: 0x0000ffffe9002400 libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const + 512
libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const:
->  0xffffe9002400 <+512>: str    q0, [x26, w0, sxtw]
    0xffffe9002404 <+516>: ldp    x1, x0, [x19, #0x8]
    0xffffe9002408 <+520>: ldr    w0, [x0]
    0xffffe900240c <+524>: ldr    w1, [x1]
  thread #27, name = 'python', stop reason = signal SIGSEGV: invalid address (fault address: 0xfff88d2330c0)
    frame #0: 0x0000ffffe9002400 libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const + 512
libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const:
->  0xffffe9002400 <+512>: str    q0, [x26, w0, sxtw]
    0xffffe9002404 <+516>: ldp    x1, x0, [x19, #0x8]
    0xffffe9002408 <+520>: ldr    w0, [x0]
    0xffffe900240c <+524>: ldr    w1, [x1]
  thread #28, name = 'python', stop reason = signal SIGSEGV: invalid address (fault address: 0xfff88d1b50c0)
    frame #0: 0x0000ffffe9002400 libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const + 512
libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const:
->  0xffffe9002400 <+512>: str    q0, [x26, w0, sxtw]
    0xffffe9002404 <+516>: ldp    x1, x0, [x19, #0x8]
    0xffffe9002408 <+520>: ldr    w0, [x0]
    0xffffe900240c <+524>: ldr    w1, [x1]
  thread #29, name = 'python', stop reason = signal SIGSEGV: invalid address (fault address: 0xfff88d1370c0)
    frame #0: 0x0000ffffe9002400 libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const + 512
libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const:
->  0xffffe9002400 <+512>: str    q0, [x26, w0, sxtw]
    0xffffe9002404 <+516>: ldp    x1, x0, [x19, #0x8]
    0xffffe9002408 <+520>: ldr    w0, [x0]
    0xffffe900240c <+524>: ldr    w1, [x1]
  thread #30, name = 'python', stop reason = signal SIGSEGV: invalid address (fault address: 0xfff88d0b90c0)
    frame #0: 0x0000ffffe9002400 libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const + 512
libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const:
->  0xffffe9002400 <+512>: str    q0, [x26, w0, sxtw]
    0xffffe9002404 <+516>: ldp    x1, x0, [x19, #0x8]
    0xffffe9002408 <+520>: ldr    w0, [x0]
    0xffffe900240c <+524>: ldr    w1, [x1]
(lldb) bt
* thread #1, name = 'python', stop reason = signal SIGSEGV: invalid address (fault address: 0xfff88d79d0c0)
  * frame #0: 0x0000ffffe9002400 libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool)::'lambda'(arm_compute::Coordinates const&)::operator()(arm_compute::Coordinates const&) const + 512
    frame #1: 0x0000ffffe9002c14 libarm_compute.so`void arm_compute::run_reverse<unsigned int>(arm_compute::Window const&, arm_compute::ITensor const*, arm_compute::ITensor const*, arm_compute::ITensor*, bool) + 1364
    frame #2: 0x0000ffffe8f4f3bc libarm_compute.so`std::_Function_handler<void (arm_compute::ThreadInfo const&), arm_compute::IScheduler::schedule_common(arm_compute::ICPPKernel*, arm_compute::IScheduler::Hints const&, arm_compute::Window const&, arm_compute::ITensorPack&)::'lambda0'(arm_compute::ThreadInfo const&)>::_M_invoke(std::_Any_data const&, arm_compute::ThreadInfo const&) + 380
    frame #3: 0x0000ffffe8f6c3d4 libarm_compute.so`arm_compute::OMPScheduler::run_workloads(std::vector<std::function<void (arm_compute::ThreadInfo const&)>, std::allocator<std::function<void (arm_compute::ThreadInfo const&)> > >&) (._omp_fn.0) + 148
    frame #4: 0x0000ffffe98e4de4 libgomp.so.1`GOMP_parallel(fn=(libarm_compute.so`arm_compute::OMPScheduler::run_workloads(std::vector<std::function<void (arm_compute::ThreadInfo const&)>, std::allocator<std::function<void (arm_compute::ThreadInfo const&)> > >&) (._omp_fn.0)), data=0x0000ffffffffb580, num_threads=16, flags=3) at parallel.c:178:3
    frame #5: 0x0000ffffe8f6c308 libarm_compute.so`arm_compute::OMPScheduler::run_workloads(std::vector<std::function<void (arm_compute::ThreadInfo const&)>, std::allocator<std::function<void (arm_compute::ThreadInfo const&)> > >&) + 152
    frame #6: 0x0000ffffe8f4feb0 libarm_compute.so`arm_compute::IScheduler::schedule_common(arm_compute::ICPPKernel*, arm_compute::IScheduler::Hints const&, arm_compute::Window const&, arm_compute::ITensorPack&) + 1232
    frame #7: 0x0000ffffe8f6c914 libarm_compute.so`arm_compute::OMPScheduler::schedule(arm_compute::ICPPKernel*, arm_compute::IScheduler::Hints const&) + 116
    frame #8: 0x0000ffffe8f70e68 libarm_compute.so`arm_compute::INESimpleFunctionNoBorder::run() + 56
    frame #9: 0x0000ffffe90735e8 libarm_compute.so`arm_compute::NEDeconvolutionLayer::run() + 244
    frame #10: 0x0000fffff3242978 libtorch_cpu.so`dnnl::impl::cpu::aarch64::acl_deconvolution_fwd_t::execute_forward(dnnl::impl::exec_ctx_t const&) const + 648
    frame #11: 0x0000fffff20235ac libtorch_cpu.so`dnnl_primitive::execute(dnnl::impl::exec_ctx_t&) const + 156
    frame #12: 0x0000fffff2024374 libtorch_cpu.so`dnnl::impl::primitive_execute(dnnl_primitive const*, dnnl::impl::exec_ctx_t&) + 1764
    frame #13: 0x0000fffff2025328 libtorch_cpu.so`dnnl_primitive_execute + 376
    frame #14: 0x0000ffffec48fce8 libtorch_cpu.so`dnnl::primitive::execute(dnnl::stream const&, std::unordered_map<int, dnnl::memory, std::hash<int>, std::equal_to<int>, std::allocator<std::pair<int const, dnnl::memory> > > const&) const + 232
    frame #15: 0x0000ffffecc70950 libtorch_cpu.so`void ideep::convolution_transpose_forward::do_compute<true, true, true>(ideep::deconv_forward_params const&, ideep::tensor const&, ideep::tensor const&, ideep::tensor const&, ideep::tensor&) + 1312
    frame #16: 0x0000ffffecd9b1d4 libtorch_cpu.so`at::native::(anonymous namespace)::_mkldnn_convolution_transpose(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, long, bool, std::basic_string_view<char, std::char_traits<char> >, c10::List<std::optional<c10::Scalar> >, std::optional<std::basic_string_view<char, std::char_traits<char> > >) + 3152
    frame #17: 0x0000ffffecd9bd30 libtorch_cpu.so`at::native::(anonymous namespace)::mkldnn_convolution_transpose(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, long) + 272
    frame #18: 0x0000ffffec571d9c libtorch_cpu.so`at::native::_convolution(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, bool, c10::ArrayRef<long>, long, bool, bool, bool, bool) + 4492
    frame #19: 0x0000ffffeda58d28 libtorch_cpu.so`c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor (at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt, bool, bool, bool, bool), &(at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeExplicitAutograd___convolution(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt, bool, bool, bool, bool))>, at::Tensor, c10::guts::typelist::typelist<at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt, bool, bool, bool, bool> >, at::Tensor (at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt, bool, bool, bool, bool)>::call(c10::OperatorKernel*, c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt, bool, bool, bool, bool) + 328
    frame #20: 0x0000ffffecf5ae10 libtorch_cpu.so`at::_ops::_convolution::call(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt, bool, bool, bool, bool) + 688
    frame #21: 0x0000ffffec5628d8 libtorch_cpu.so`at::native::convolution(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, bool, c10::ArrayRef<long>, long) + 420
    frame #22: 0x0000ffffeda59060 libtorch_cpu.so`c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor (at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt), &(at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeExplicitAutograd__convolution(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt))>, at::Tensor, c10::guts::typelist::typelist<at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt> >, at::Tensor (at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt)>::call(c10::OperatorKernel*, c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt) + 272
    frame #23: 0x0000ffffecef62f4 libtorch_cpu.so`at::_ops::convolution::redispatch(c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt) + 320
    frame #24: 0x0000fffff0107ff8 libtorch_cpu.so`torch::autograd::VariableType::(anonymous namespace)::convolution(c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt) + 436
    frame #25: 0x0000fffff0108db4 libtorch_cpu.so`c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor (c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt), &(torch::autograd::VariableType::(anonymous namespace)::convolution(c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt))>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt> >, at::Tensor (c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt)>::call(c10::OperatorKernel*, c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt) + 128
    frame #26: 0x0000ffffecf59fa8 libtorch_cpu.so`at::_ops::convolution::call(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt) + 616
    frame #27: 0x0000ffffec5644b0 libtorch_cpu.so`at::native::conv_transpose1d_symint(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt, c10::ArrayRef<c10::SymInt>) + 428
    frame #28: 0x0000ffffedbc16d0 libtorch_cpu.so`c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor (at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt, c10::ArrayRef<c10::SymInt>), &(at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeImplicitAutograd__conv_transpose1d(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt, c10::ArrayRef<c10::SymInt>))>, at::Tensor, c10::guts::typelist::typelist<at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt, c10::ArrayRef<c10::SymInt> > >, at::Tensor (at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt, c10::ArrayRef<c10::SymInt>)>::call(c10::OperatorKernel*, c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt, c10::ArrayRef<c10::SymInt>) + 204
    frame #29: 0x0000ffffecf5bc5c libtorch_cpu.so`at::_ops::conv_transpose1d::call(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt, c10::ArrayRef<c10::SymInt>) + 600
    frame #30: 0x0000fffff61b80f8 libtorch_python.so`torch::autograd::THPVariable_conv_transpose1d(_object*, _object*, _object*) + 916
    frame #31: 0x0000aaaaaacb8edc python`cfunction_call + 140
    frame #32: 0x0000aaaaaab1c3c0 python`_PyObject_MakeTpCall + 160
    frame #33: 0x0000aaaaaab0c240 python`_PyEval_EvalFrameDefault + 27312
    frame #34: 0x0000aaaaaabb7460 python`_PyEval_Vector + 80
    frame #35: 0x0000aaaaaaca0730 python`method_vectorcall + 220
    frame #36: 0x0000aaaaaab06b0c python`_PyEval_EvalFrameDefault + 4988
    frame #37: 0x0000aaaaaabb7460 python`_PyEval_Vector + 80
    frame #38: 0x0000aaaaaaca0730 python`method_vectorcall + 220
    frame #39: 0x0000aaaaaab06b0c python`_PyEval_EvalFrameDefault + 4988
    frame #40: 0x0000aaaaaabb7460 python`_PyEval_Vector + 80
    frame #41: 0x0000aaaaaab1c5a8 python`_PyObject_FastCallDictTstate + 104
    frame #42: 0x0000aaaaaab1c8ec python`_PyObject_Call_Prepend + 268
    frame #43: 0x0000aaaaaab6e440 python`slot_tp_call + 240
    frame #44: 0x0000aaaaaab1c3c0 python`_PyObject_MakeTpCall + 160
    frame #45: 0x0000aaaaaab0cf10 python`_PyEval_EvalFrameDefault + 30592
    frame #46: 0x0000aaaaaabb7460 python`_PyEval_Vector + 80
    frame #47: 0x0000aaaaaabb7654 python`PyEval_EvalCode + 144
    frame #48: 0x0000aaaaaabf6bbc python`run_eval_code_obj + 108
    frame #49: 0x0000aaaaaabf6e58 python`run_mod + 132
    frame #50: 0x0000aaaaaabf9de0 python`PyRun_StringFlags + 160
    frame #51: 0x0000aaaaaabf9e88 python`PyRun_SimpleStringFlags + 88
    frame #52: 0x0000aaaaaab0f18c python`Py_RunMain + 380
    frame #53: 0x0000aaaaaab0fec4 python`Py_BytesMain + 100
    frame #54: 0x0000fffff7d37400 libc.so.6`__libc_start_call_main(main=(python`main), argc=3, argv=0x0000ffffffffec18) at libc_start_call_main.h:58:16
    frame #55: 0x0000fffff7d374d8 libc.so.6`__libc_start_main_impl(main=(python`main), argc=3, argv=0x0000ffffffffec18, init=(python`__libc_csu_init), fini=<unavailable>, rtld_fini=<unavailable>, stack_end=<unavailable>) at libc-start.c:392:3
    frame #56: 0x0000aaaaaab0e6b8 python`_start + 56