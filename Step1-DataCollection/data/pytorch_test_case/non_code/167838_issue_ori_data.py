Traceback (most recent call last):
  File "/my_code/train.py", line 1473, in <module>
    main(exit_stack)
  File "/my_conda_env/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 362, in wrapper
    return f(*args, **kwargs)
  File "/my_code/train.py", line 535, in main
    profiler.step(iteration)
  File "/my_code/profiling.py", line 73, in step
    return super().step()
  File "/my_conda_env/lib/python3.10/site-packages/torch/profiler/profiler.py", line 844, in step
    self._transit_action(prev_action, self.current_action)
  File "/my_conda_env/lib/python3.10/site-packages/torch/profiler/profiler.py", line 878, in _transit_action
    action()
  File "/my_conda_env/lib/python3.10/site-packages/torch/profiler/profiler.py", line 872, in _trace_ready
    self.on_trace_ready(self)
  File "/my_code/profiling.py", line 78, in _on_trace_ready
    self._on_trace_ready_orig(prof, self._iteration)
  File "/my_code/profiling.py", line 114, in trace_handler
    prof.export_memory_timeline(p.as_posix())
  File "/my_conda_env/lib/python3.10/site-packages/torch/profiler/profiler.py", line 431, in export_memory_timeline
    self.mem_tl = MemoryProfileTimeline(self._memory_profile())
  File "/my_conda_env/lib/python3.10/site-packages/torch/profiler/profiler.py", line 399, in _memory_profile
    return MemoryProfile(self.profiler.kineto_results)
  File "/my_conda_env/lib/python3.10/site-packages/torch/profiler/_memory_profiler.py", line 658, in __init__
    self._data_flow_graph = DataFlowGraph(self._op_tree)
  File "/my_conda_env/lib/python3.10/site-packages/torch/profiler/_memory_profiler.py", line 511, in __init__
    self._flow_nodes = [DataFlowNode(e, self) for e in self.leaf_events]
  File "/my_conda_env/lib/python3.10/site-packages/torch/profiler/_memory_profiler.py", line 511, in <listcomp>
    self._flow_nodes = [DataFlowNode(e, self) for e in self.leaf_events]
  File "/my_conda_env/lib/python3.10/site-packages/torch/profiler/_memory_profiler.py", line 424, in __init__
    assert all(i == j for i, j in versions.values()), f"{versions}, {self._edges}"
AssertionError: {id=11:     0xffd1a8000000 (12)  (cuda:1): (0, 2)}, {id=11:     0xffd1a8000000 (12)  (cuda:1): DataFlowEdge(input_version=None, mutated=False)}