SerializeError: Failed serializing node mul_1831 in graph: %mul_1831 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, 1j), kwargs = {})
 Original exception Traceback (most recent call last):
  File "/fsx/home/srdecny/meaning4/env/lib/python3.11/site-packages/torch/_export/serde/serialize.py", line 1547, in serialize_graph
    getattr(self, f"handle_{node.op}")(node)
  File "/fsx/home/srdecny/meaning4/env/lib/python3.11/site-packages/torch/_export/serde/serialize.py", line 563, in handle_call_function
    inputs=self.serialize_inputs(node.target, node.args, node.kwargs),
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/fsx/home/srdecny/meaning4/env/lib/python3.11/site-packages/torch/_export/serde/serialize.py", line 778, in serialize_inputs
    arg=self.serialize_input(args[i], schema_arg.type),
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/fsx/home/srdecny/meaning4/env/lib/python3.11/site-packages/torch/_export/serde/serialize.py", line 1096, in serialize_input
    raise SerializeError(
torch._export.serde.serialize.SerializeError: Unsupported argument type: <class 'complex'> with schema arg_type Tensor