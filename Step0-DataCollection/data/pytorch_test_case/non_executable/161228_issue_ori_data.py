% lldb -- python ../test/test_mps.py -v -k lkjhkjhjkh
(lldb) target create "python"
Current executable set to '/Users/nshulga/py3.12-dummy/bin/python' (arm64).
(lldb) settings set -- target.run-args  "../test/test_mps.py" "-v" "-k" "lkjhkjhjkh"
(lldb) r
Process 88862 launched: '/Users/nshulga/py3.12-dummy/bin/python' (arm64)
Process 88862 stopped
* thread #2, stop reason = exec
    frame #0: 0x00000001000147c0 dyld`_dyld_start
dyld`_dyld_start:
->  0x1000147c0 <+0>:  mov    x0, sp
    0x1000147c4 <+4>:  and    sp, x0, #0xfffffffffffffff0
    0x1000147c8 <+8>:  mov    x29, #0x0 ; =0 
    0x1000147cc <+12>: mov    x30, #0x0 ; =0 
(lldb) c
Process 88862 resuming
Process 88862 stopped
* thread #2, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x762f6863726f7479)
    frame #0: 0x0000000195bc83b8 libsystem_platform.dylib`_platform_strcmp$VARIANT$Base + 8
libsystem_platform.dylib`_platform_strcmp$VARIANT$Base:
->  0x195bc83b8 <+8>:  ldrb   w4, [x0], #0x1
    0x195bc83bc <+12>: ldrb   w5, [x1], #0x1
    0x195bc83c0 <+16>: subs   x3, x4, x5
    0x195bc83c4 <+20>: ccmp   w4, #0x0, #0x4, eq
(lldb) bt
* thread #2, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x762f6863726f7479)
  * frame #0: 0x0000000195bc83b8 libsystem_platform.dylib`_platform_strcmp$VARIANT$Base + 8
    frame #1: 0x0000000148341378 libtorch_cpu.dylib`c10::impl::OperatorEntry::registerKernel(c10::Dispatcher const&, std::__1::optional<c10::DispatchKey>, c10::KernelFunction, std::__1::optional<c10::impl::CppSignature>, std::__1::unique_ptr<c10::FunctionSchema, std::__1::default_delete<c10::FunctionSchema>>, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>) + 172
    frame #2: 0x000000014833c080 libtorch_cpu.dylib`c10::Dispatcher::registerImpl(c10::OperatorName, std::__1::optional<c10::DispatchKey>, c10::KernelFunction, std::__1::optional<c10::impl::CppSignature>, std::__1::unique_ptr<c10::FunctionSchema, std::__1::default_delete<c10::FunctionSchema>>, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>) + 192
    frame #3: 0x0000000148372afc libtorch_cpu.dylib`torch::Library::_impl(char const*, torch::CppFunction&&, torch::_RegisterOrVerify) & + 432
    frame #4: 0x000000012c5c3dbc _C.so`vision::ops::TORCH_LIBRARY_IMPL_init_torchvision_CPU_2(torch::Library&) + 316
    frame #5: 0x000000012c573c60 _C.so`torch::detail::TorchLibraryInit::TorchLibraryInit(torch::Library::Kind, void (*)(torch::Library&), char const*, std::__1::optional<c10::DispatchKey>, char const*, unsigned int) + 216
    frame #6: 0x000000012c5d11b0 _C.so`OUTLINED_FUNCTION_3 + 80
    frame #7: 0x0000000195806efc dyld`invocation function for block in dyld4::Loader::findAndRunAllInitializers(dyld4::RuntimeState&) const + 444
    frame #8: 0x0000000195843678 dyld`invocation function for block in dyld3::MachOAnalyzer::forEachInitializer(Diagnostics&, dyld3::MachOAnalyzer::VMAddrConverter const&, void (unsigned int) block_pointer, void const*) const + 172
    frame #9: 0x00000001958635cc dyld`invocation function for block in mach_o::Header::forEachSection(void (mach_o::Header::SectionInfo const&, bool&) block_pointer) const + 240
    frame #10: 0x0000000195860358 dyld`mach_o::Header::forEachLoadCommand(void (load_command const*, bool&) block_pointer) const + 208
    frame #11: 0x0000000195861a98 dyld`mach_o::Header::forEachSection(void (mach_o::Header::SectionInfo const&, bool&) block_pointer) const + 124
    frame #12: 0x0000000195838ba0 dyld`dyld3::MachOFile::forEachInitializerPointerSection(Diagnostics&, void (unsigned int, unsigned int, bool&) block_pointer) const + 160
    frame #13: 0x0000000195843318 dyld`dyld3::MachOAnalyzer::forEachInitializer(Diagnostics&, dyld3::MachOAnalyzer::VMAddrConverter const&, void (unsigned int) block_pointer, void const*) const + 432
    frame #14: 0x0000000195806cb4 dyld`dyld4::Loader::findAndRunAllInitializers(dyld4::RuntimeState&) const + 176
    frame #15: 0x000000019580e670 dyld`dyld4::JustInTimeLoader::runInitializers(dyld4::RuntimeState&) const + 36
    frame #16: 0x0000000195807460 dyld`dyld4::Loader::runInitializersBottomUp(dyld4::RuntimeState&, dyld3::Array<dyld4::Loader const*>&, dyld3::Array<dyld4::Loader const*>&) const + 308
    frame #17: 0x000000019580bbf0 dyld`dyld4::Loader::runInitializersBottomUpPlusUpwardLinks(dyld4::RuntimeState&) const::$_0::operator()() const + 180
    frame #18: 0x000000019580777c dyld`dyld4::Loader::runInitializersBottomUpPlusUpwardLinks(dyld4::RuntimeState&) const + 716
    frame #19: 0x000000019582b144 dyld`dyld4::APIs::dlopen_from(char const*, int, void*)::$_0::operator()() const + 1856
    frame #20: 0x000000019581ff9c dyld`dyld4::APIs::dlopen_from(char const*, int, void*) + 1136
    frame #21: 0x000000019581fa80 dyld`dyld4::APIs::dlopen(char const*, int) + 128
    frame #22: 0x0000000101022730 _ctypes.cpython-312-darwin.so`py_dl_open + 136
    frame #23: 0x00000001007abaa0 Python`cfunction_call + 108
    frame #24: 0x0000000100759b08 Python`_PyObject_MakeTpCall + 124
    frame #25: 0x000000010084cf40 Python`_PyEval_EvalFrameDefault + 23304
    frame #26: 0x0000000100759934 Python`_PyObject_FastCallDictTstate + 92
    frame #27: 0x00000001007d095c Python`slot_tp_init + 212
    frame #28: 0x00000001007c7b60 Python`type_call + 148
    frame #29: 0x0000000100759b08 Python`_PyObject_MakeTpCall + 124
    frame #30: 0x000000010084cf40 Python`_PyEval_EvalFrameDefault + 23304
    frame #31: 0x00000001008471c8 Python`PyEval_EvalCode + 184
    frame #32: 0x0000000100843440 Python`builtin_exec + 444
    frame #33: 0x00000001007ab1b4 Python`cfunction_vectorcall_FASTCALL_KEYWORDS + 92
    frame #34: 0x0000000100848808 Python`_PyEval_EvalFrameDefault + 5072
    frame #35: 0x000000010075bc44 Python`object_vacall + 228
    frame #36: 0x000000010075bb14 Python`PyObject_CallMethodObjArgs + 104
    frame #37: 0x00000001008855fc Python`PyImport_ImportModuleLevelObject + 1180
    frame #38: 0x000000010084af30 Python`_PyEval_EvalFrameDefault + 15096
    frame #39: 0x00000001008471c8 Python`PyEval_EvalCode + 184
    frame #40: 0x0000000100843440 Python`builtin_exec + 444
    frame #41: 0x00000001007ab1b4 Python`cfunction_vectorcall_FASTCALL_KEYWORDS + 92
    frame #42: 0x0000000100848808 Python`_PyEval_EvalFrameDefault + 5072
    frame #43: 0x000000010075bc44 Python`object_vacall + 228
    frame #44: 0x000000010075bb14 Python`PyObject_CallMethodObjArgs + 104
    frame #45: 0x00000001008855fc Python`PyImport_ImportModuleLevelObject + 1180
    frame #46: 0x000000010084af30 Python`_PyEval_EvalFrameDefault + 15096
    frame #47: 0x00000001008471c8 Python`PyEval_EvalCode + 184
    frame #48: 0x0000000100843440 Python`builtin_exec + 444
    frame #49: 0x00000001007ab1b4 Python`cfunction_vectorcall_FASTCALL_KEYWORDS + 92
    frame #50: 0x0000000100848808 Python`_PyEval_EvalFrameDefault + 5072
    frame #51: 0x000000010075bc44 Python`object_vacall + 228
    frame #52: 0x000000010075bb14 Python`PyObject_CallMethodObjArgs + 104
    frame #53: 0x00000001008855fc Python`PyImport_ImportModuleLevelObject + 1180
    frame #54: 0x000000010084af30 Python`_PyEval_EvalFrameDefault + 15096
    frame #55: 0x00000001008471c8 Python`PyEval_EvalCode + 184
    frame #56: 0x00000001008a78bc Python`run_eval_code_obj + 88
    frame #57: 0x00000001008a5994 Python`run_mod + 132
    frame #58: 0x00000001008a4e08 Python`pyrun_file + 156
    frame #59: 0x00000001008a416c Python`_PyRun_SimpleFileObject + 288
    frame #60: 0x00000001008a3d64 Python`_PyRun_AnyFileObject + 80
    frame #61: 0x00000001008ca194 Python`pymain_run_file_obj + 164
    frame #62: 0x00000001008c9f08 Python`pymain_run_file + 72
    frame #63: 0x00000001008c94ec Python`Py_RunMain + 852
    frame #64: 0x00000001008c9950 Python`pymain_main + 304
    frame #65: 0x00000001008c99f0 Python`Py_BytesMain + 40
    frame #66: 0x00000001957eab98 dyld`start + 6076