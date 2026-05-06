File "/Applications/ComfyUI.app/Contents/Resources/ComfyUI/execution.py", line 496, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, hidden_inputs=hidden_inputs)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/Applications/ComfyUI.app/Contents/Resources/ComfyUI/execution.py", line 315, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, hidden_inputs=hidden_inputs)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/Applications/ComfyUI.app/Contents/Resources/ComfyUI/execution.py", line 289, in _async_map_node_over_list
    await process_inputs(input_dict, i)

  File "/Applications/ComfyUI.app/Contents/Resources/ComfyUI/execution.py", line 277, in process_inputs
    result = f(**inputs)
             ^^^^^^^^^^^

  File "/Users/rodrigo/Documents/ComfyUI/custom_nodes/comfyui-advancedliveportrait/nodes.py", line 947, in run
    crop_out = pipeline.warp_decode(psi.f_s_user, psi.x_s_user, x_d_new)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/Users/rodrigo/Documents/ComfyUI/custom_nodes/comfyui-advancedliveportrait/LivePortrait/live_portrait_wrapper.py", line 130, in warp_decode
    ret_dct = self.warping_module(feature_3d, kp_source=kp_source, kp_driving=kp_driving)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/Users/rodrigo/Documents/ComfyUI/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/Users/rodrigo/Documents/ComfyUI/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/Users/rodrigo/Documents/ComfyUI/custom_nodes/comfyui-advancedliveportrait/LivePortrait/modules/warping_network.py", line 52, in forward
    dense_motion = self.dense_motion_network(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/Users/rodrigo/Documents/ComfyUI/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/Users/rodrigo/Documents/ComfyUI/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/Users/rodrigo/Documents/ComfyUI/custom_nodes/comfyui-advancedliveportrait/LivePortrait/modules/dense_motion.py", line 78, in forward
    deformed_feature = self.create_deformed_feature(feature, sparse_motion)  # (bs, 1+num_kp, c=4, d=16, h=64, w=64)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/Users/rodrigo/Documents/ComfyUI/custom_nodes/comfyui-advancedliveportrait/LivePortrait/modules/dense_motion.py", line 50, in create_deformed_feature
    sparse_deformed = F.grid_sample(feature_repeat, sparse_motions, align_corners=False)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/Users/rodrigo/Documents/ComfyUI/.venv/lib/python3.12/site-packages/torch/nn/functional.py", line 5108, in grid_sample
    return torch.grid_sampler(input, grid, mode_enum, padding_mode_enum, align_corners)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^