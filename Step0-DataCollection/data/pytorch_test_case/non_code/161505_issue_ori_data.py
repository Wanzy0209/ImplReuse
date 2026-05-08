==================================================================== FAILURES =====================================================================
__________________________________________ BeitImageProcessingTest.test_can_compile_fast_image_processor __________________________________________

self = <tests.models.beit.test_image_processing_beit.BeitImageProcessingTest testMethod=test_can_compile_fast_image_processor>

    @slow
    @require_torch_accelerator
    @require_vision
    @pytest.mark.torch_compile_test
    def test_can_compile_fast_image_processor(self):
        if self.fast_image_processing_class is None:
            self.skipTest("Skipping compilation test as fast image processor is not defined")
        if version.parse(torch.__version__) < version.parse("2.3"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        torch.compiler.reset()
        input_image = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)
        image_processor = self.fast_image_processing_class(**self.image_processor_dict)
        output_eager = image_processor(input_image, device=torch_device, return_tensors="pt")

        image_processor = torch.compile(image_processor, mode="reduce-overhead")
>       output_compiled = image_processor(input_image, device=torch_device, return_tensors="pt")

tests/test_image_processing_common.py:631:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
/usr/local/lib/python3.10/dist-packages/torch/_dynamo/eval_frame.py:807: in compile_wrapper
    return fn(*args, **kwargs)
/usr/local/lib/python3.10/dist-packages/torch/_dynamo/external_utils.py:68: in inner
    return fn(*args, **kwargs)
src/transformers/image_processing_utils_fast.py:627: in __call__
    return self.preprocess(images, *args, **kwargs)
src/transformers/models/beit/image_processing_beit_fast.py:104: in preprocess
    return super().preprocess(images, segmentation_maps, **kwargs)
src/transformers/image_processing_utils_fast.py:663: in preprocess
    return self._preprocess_image_like_inputs(
src/transformers/models/beit/image_processing_beit_fast.py:118: in _preprocess_image_like_inputs
    images = self._prepare_image_like_inputs(
/usr/local/lib/python3.10/dist-packages/torch/_dynamo/convert_frame.py:1856: in __call__
    result = self._torchdynamo_orig_backend(
/usr/local/lib/python3.10/dist-packages/torch/_dynamo/convert_frame.py:1609: in __call__
    result = self._inner_convert(
/usr/local/lib/python3.10/dist-packages/torch/_dynamo/convert_frame.py:693: in __call__
    result = _compile(
/usr/local/lib/python3.10/dist-packages/torch/_dynamo/convert_frame.py:1423: in _compile
    guarded_code, tracer_output = compile_inner(code, one_graph, hooks)
/usr/local/lib/python3.10/dist-packages/torch/_utils_internal.py:92: in wrapper_function
    return function(*args, **kwargs)
/usr/local/lib/python3.10/dist-packages/torch/_dynamo/convert_frame.py:1104: in compile_inner
    return _compile_inner(code, one_graph, hooks)
/usr/local/lib/python3.10/dist-packages/torch/_dynamo/convert_frame.py:1138: in _compile_inner
    dynamo_output = compile_frame(
/usr/local/lib/python3.10/dist-packages/torch/_dynamo/convert_frame.py:1019: in compile_frame
    bytecode, tracer_output = transform_code_object(code, transform)
/usr/local/lib/python3.10/dist-packages/torch/_dynamo/bytecode_transformation.py:1592: in transform_code_object
    tracer_output = transformations(instructions, code_options)
/usr/local/lib/python3.10/dist-packages/torch/_dynamo/convert_frame.py:991: in transform
    tracer_output = trace_frame(
/usr/local/lib/python3.10/dist-packages/torch/_dynamo/convert_frame.py:311: in _fn
    return fn(*args, **kwargs)
/usr/local/lib/python3.10/dist-packages/torch/_dynamo/convert_frame.py:820: in trace_frame
    run_tracer()
/usr/local/lib/python3.10/dist-packages/torch/_dynamo/convert_frame.py:802: in run_tracer
    tracer.run()
/usr/local/lib/python3.10/dist-packages/torch/_dynamo/symbolic_convert.py:1472: in run
    while self.step():
/usr/local/lib/python3.10/dist-packages/torch/_dynamo/symbolic_convert.py:1342: in step
    self.dispatch_table[inst.opcode](self, inst)
/usr/local/lib/python3.10/dist-packages/torch/_dynamo/symbolic_convert.py:4039: in RETURN_VALUE
    self._return(inst)
/usr/local/lib/python3.10/dist-packages/torch/_dynamo/symbolic_convert.py:4017: in _return
    all_stack_locals_metadata = self.output.compile_subgraph(
/usr/local/lib/python3.10/dist-packages/torch/_dynamo/output_graph.py:1475: in compile_subgraph
    self.codegen_suffix(tx, stack_values_flat, pass1)
/usr/local/lib/python3.10/dist-packages/torch/_dynamo/output_graph.py:1657: in codegen_suffix
    self.side_effects.codegen_save_tempvars(cg)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <torch._dynamo.side_effects.SideEffects object at 0x7f41f24d2e90>, cg = <torch._dynamo.codegen.PyCodegen object at 0x7f41f13c1120>

    def codegen_save_tempvars(self, cg: PyCodegen) -> None:
        # We must codegen modified VT to their source by default, so that
        # mutation and aliasing are properly accounted for.
        #
        # Since newly constructed objects don't have a source, we manually
        # codegen their construction and store them to a newly assigned local
        # source. Note that `ValueMutationNew` isn't tracked by SideEffects.
        for var in self._get_modified_vars():
            if not isinstance(var.mutation_type, AttributeMutationNew):
                assert var.source is not None
                continue

            if isinstance(var, variables.CellVariable):
                # Cells created in the root frame are created either by
                # `MAKE_CELL` or by them being in `co_cellvars`, so we only emit
                # `make_cell` for the non-root-frame cells here.
                # TODO generalize this so we never need to call `make_cell`.
                if var.local_name is None:
                    cg.add_push_null(
                        lambda: cg.load_import_from(utils.__name__, "make_cell")
                    )
                    cg.extend_output(create_call_function(0, False))
                    cg.add_cache(var)
                    var.source = LocalSource(cg.tempvars[var])  # type: ignore[attr-defined]
                elif var.source is None:
                    var.source = LocalCellSource(var.local_name)
            elif isinstance(var, variables.TensorVariable):
                # NOTE: for historical reasons we never assigned local sources
                # to newly constructed tensor object, so we keep it that way.
                # They are always loaded from output of the fx graph, so one can
                # think of it as having a "OutputGraphSource" for codegen
                # purposes.
                #
                # However, tensor subclass objects are different, because the
                # reconstruction logic in `PyCodegen` loads the data tensor from
                # graph output and then calls `as_subclass`, meaning we must
                # assign a source to it to ensure we only reconstruct one
                # subclass instance.
                if isinstance(
                    var, variables.torch_function.TensorWithTFOverrideVariable
                ):
                    # Don't codegen from temp source assigned from the 1st pass.
                    cg(var, allow_cache=False)
                    cg.add_cache(var)
                    # `add_cache` generates STORE and consumes TOS, but we never
                    # cleared it. TODO move this call into `add_cache`
                    cg.clear_tos()
                    var.source = LocalSource(cg.tempvars[var])
            elif isinstance(var, variables.AutogradFunctionContextVariable):
                unimplemented_v2(
                    gb_type="AutogradFunctionContextVariable escaped Dynamo-traced region",
                    context="",
                    explanation="We cannot reconstruct a torch.autograd.Function's context object.",
                    hints=[],
                )
            else:
                # Reconstruct the bytecode for
                # base_cls.__new__(user_cls, *args)
                if isinstance(var, variables.UserDefinedObjectVariable):

                    def load_new_method() -> None:
                        assert var.base_cls_vt is not None
                        cg(var.base_cls_vt)  # type: ignore[attr-defined]
                        cg.extend_output([cg.create_load_attr("__new__")])

                    cg.add_push_null(load_new_method)
                else:
                    cg.add_push_null(
                        lambda: cg.load_import_from(utils.__name__, "object_new")
                    )
>               assert var.mutation_type.cls_source is not None
E               AssertionError:
E
E               from user code:
E                  File "/transformers/src/transformers/models/beit/image_processing_beit_fast.py", line 140, in torch_dynamo_resume_in__preprocess_image_like_inputs_at_118
E                   return batch_feature
E
E               Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

/usr/local/lib/python3.10/dist-packages/torch/_dynamo/side_effects.py:744: AssertionError