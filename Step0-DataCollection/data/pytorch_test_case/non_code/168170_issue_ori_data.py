Traceback (most recent call last):
  File "/var/lib/jenkins/workspace/test/inductor/test_memory_planning.py", line 140, in test_unbacked_symint
    ).run(code)
RuntimeError: Expected to find "const int64_t int_array_2[] = {10L, 8L*u0, 32L};" but did not find it
Searched string:
    int32_t u0_raw;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_int32(buf0, &u0_raw));
    auto u0 = u0_raw;
    pool1.reset(); buf0.reset();
    if (!(u0 >= 1L)) { throw std::runtime_error("Expected u0 >= 1 but received " + std::to_string(u0)); }
    if (!(1L <= u0)) { throw std::runtime_error("Expected 1 <= u0 but received " + std::to_string(u0)); }
    if (!(u0 >= 1L)) { throw std::runtime_error("Expected u0 >= 1 to be True but received " + std::to_string(u0)); }

    const int64_t int_array_3[] = {10L, 8L*u0, 32L};
    const int64_t int_array_4[] = {256L*u0, 32L, 1L};
    AtenTensorHandle pool0_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(3, int_array_3, int_array_4, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &pool0_handle));
    RAIIAtenTensorHandle pool0(pool0_handle);
    const int64_t int_array_5[] = {10L, 8L*u0, 32L};
    const int64_t int_array_6[] = {256L*u0, 32L, 1L};
    AtenTensorHandle tmp_tensor_handle_1;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch__alloc_from_pool(pool0, 0, cached_torch_dtype_float32, 3, int_array_5, int_array_6, &tmp_tensor_handle_1));
    auto buf5 = RAIIAtenTensorHandle(tmp_tensor_handle_1);
    // Topologically Sorted Source Nodes: [fill_, mul_2], Original ATen: [aten.fill, aten.mul]
    int64_t triton_poi_fused_fill_mul_1_xnumel = 2560L*u0;
    call_triton_poi_fused_fill_mul_1(buf5, triton_poi_fused_fill_mul_1_xnumel, this->device_idx_, stream, kernels, this->cubin_dir_);
    output_handles[0] = buf5.release();
} // AOTInductorModel::run_impl
} // namespace torch::aot_inductor




Wrapper code written to: /tmp/tmphch7xeq7/cgpzuvaqovcgu2cfnmq4lduy3y3v5jrpsv25z52bm7ghrgdpu4wq/cxj4a5oicrncs6adnwogizrfbyr2dkhurdsmqw3g5my7w3qfeeo4.wrapper.cpp
Kernel code written to: /tmp/tmphch7xeq7/cgpzuvaqovcgu2cfnmq4lduy3y3v5jrpsv25z52bm7ghrgdpu4wq/cmusjvgfm4yaonr5dbpmj6lotvvmykonhjsv6aecqe2yunfkrdfy.kernel.cpp
From CHECK: const int64_t int_array_2[] = {10L, 8L*u0, 32L};


To execute this test, run the following from the base repo dir:
    PYTORCH_TEST_CUDA_MEM_LEAK_CHECK=1 PYTORCH_TEST_WITH_SLOW_GRADCHECK=1 python test/inductor/test_memory_planning.py TestMemoryPlanning.test_unbacked_symint

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0