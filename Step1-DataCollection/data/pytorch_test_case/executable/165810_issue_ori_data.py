class inner_f(torch.nn.Module):
    def forward(self, primals, tangents):
        primals_1: "f32[256, 256]"; primals_2: "f32[15, 256]"; tangents_1: "f32[15, 256]"; 
    
        primals_1, primals_2, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
        # File: /home/shangdiy/test_module_a2a_exact_op.py:82 in forward, code: routed_output = self.projection(x)
        t: "f32[256, 256]" = torch.ops.aten.t.default(primals_1);  primals_1 = None
        mm: "f32[15, 256]" = torch.ops.aten.mm.default(primals_2, t);  primals_2 = t = None
        
        # File: /home/shangdiy/test_module_a2a_exact_op.py:27 in _unpermute, code: out_unpermuted = out.new_empty(input_shape)
        new_empty: "f32[16, 256]" = torch.ops.aten.new_empty.default(mm, [16, 256], pin_memory = False)
        
        # File: /home/shangdiy/test_module_a2a_exact_op.py:28 in _unpermute, code: out_unpermuted[permuted_indices, :] = out
        _tensor_constant0: "i64[15]" = self._tensor_constant0
        index_put: "f32[16, 256]" = torch.ops.aten.index_put.default(new_empty, [_tensor_constant0], mm);  new_empty = _tensor_constant0 = mm = None
        
        # Annotation: {'EP': 'combine'} File: /data/users/shangdiy/pytorch/torch/distributed/_functional_collectives.py:484 in all_to_all_single, code: tensor = torch.ops._c10d_functional.all_to_all_single(  # type: ignore[attr-defined]
        slice_2: "f32[15, 256]" = torch.ops.aten.slice.Tensor(index_put, 0, 0, -1);  index_put = None
        all_to_all_single: "f32[15, 256]" = torch.ops._c10d_functional.all_to_all_single.default(slice_2, [4, 4, 4, 3], [4, 4, 4, 3], '0');  slice_2 = None
        
        # Annotation: {'EP': 'combine'} File: /data/users/shangdiy/pytorch/torch/distributed/_functional_collectives.py:135 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor: "f32[15, 256]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single);  all_to_all_single = None
        return pytree.tree_unflatten([wait_tensor, None, None], self._out_spec)