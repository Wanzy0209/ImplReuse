#0  __pthread_kill_implementation (threadid=<optimized out>, signo=signo@entry=6, no_tid=no_tid@entry=0) at pthread_kill.c:44
#1  0x00007f575188bdb3 in __pthread_kill_internal (signo=6, threadid=<optimized out>) at pthread_kill.c:78
#2  0x00007f575183eb46 in __GI_raise (sig=sig@entry=6) at ../sysdeps/posix/raise.c:26
#3  0x00007f5751828833 in __GI_abort () at abort.c:79
#4  0x00007f56d7d5635a in __cxxabiv1::__terminate (handler=<optimized out>) at /opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc/libstdc++-v3/libsupc++/eh_terminate.cc:48
#5  0x00007f56d7d553b9 in __cxa_call_terminate (ue_header=0x7f56180075b0) at /opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc/libstdc++-v3/libsupc++/eh_call.cc:54
#6  0x00007f56d7d55ae7 in __cxxabiv1::__gxx_personality_v0 (version=<optimized out>, actions=6, exception_class=5138137972254386944, ue_header=0x7f56180075b0, context=<optimized out>) at /opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc/libstdc++-v3/libsupc++/eh_personality.cc:685
#7  0x00007f56d7c9c1e4 in _Unwind_RaiseException_Phase2 (exc=exc@entry=0x7f56180075b0, context=context@entry=0x7f5637f37fe0, frames_p=frames_p@entry=0x7f5637f380d0) at /opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc/libgcc/unwind.inc:64
#8  0x00007f56d7c9c881 in _Unwind_RaiseException (exc=0x7f56180075b0) at /opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc/libgcc/unwind.inc:136
#9  0x00007f56d7d5664b in __cxxabiv1::__cxa_throw (obj=<optimized out>, tinfo=0x7f56d7eab220 <typeinfo for std::runtime_error>, dest=0x7f56d7d6b110 <std::runtime_error::~runtime_error()>) at /opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc/libstdc++-v3/libsupc++/eh_throw.cc:90
#10 0x00007f57375303a3 in dynolog::ipcfabric::FabricManager::recv() () from /home/jakobjohnson/.conda/envs/prof_repro/lib/python3.10/site-packages/torch/lib/libtorch_cpu.so
#11 0x00007f573752d7fc in libkineto::IpcFabricConfigClient::getLibkinetoOndemandConfig[abi:cxx11](int) () from /home/jakobjohnson/.conda/envs/prof_repro/lib/python3.10/site-packages/torch/lib/libtorch_cpu.so
#12 0x00007f573752bd68 in libkineto::DaemonConfigLoader::readOnDemandConfig[abi:cxx11](bool, bool) () from /home/jakobjohnson/.conda/envs/prof_repro/lib/python3.10/site-packages/torch/lib/libtorch_cpu.so
#13 0x00007f5737527aa4 in libkineto::ConfigLoader::readOnDemandConfigFromDaemon[abi:cxx11](std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::duration<long, std::ratio<1l, 1000000000l> > >) () from /home/jakobjohnson/.conda/envs/prof_repro/lib/python3.10/site-packages/torch/lib/libtorch_cpu.so
#14 0x00007f57375285da in libkineto::ConfigLoader::configureFromDaemon(std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::duration<long, std::ratio<1l, 1000000000l> > >, libkineto::Config&) () from /home/jakobjohnson/.conda/envs/prof_repro/lib/python3.10/site-packages/torch/lib/libtorch_cpu.so
#15 0x00007f5737529c73 in libkineto::ConfigLoader::updateConfigThread() () from /home/jakobjohnson/.conda/envs/prof_repro/lib/python3.10/site-packages/torch/lib/libtorch_cpu.so
#16 0x00007f56d7d80bf4 in std::execute_native_thread_routine (__p=0x5cf85c0) at /opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc/libstdc++-v3/src/c++11/thread.cc:82
#17 0x00007f575188a002 in start_thread (arg=<optimized out>) at pthread_create.c:443
#18 0x00007f575190f070 in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:81