# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import functools
import traceback

import torch
from torch._inductor.pattern_matcher import PatternMatcherPass, register_replacement, fwd_only
from torch._higher_order_ops.auto_functionalize import auto_functionalized

torch.set_default_device("cuda")

# This would be the ideal form for pattern/replacements in vLLM.
class ReluSumPattern:
    def __init__(self, e: float):
        self.e = e

    def pattern(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        return x.pow(self.e) + y.pow(self.e) + z.pow(self.e)

    def replacement(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        return (x + y + z).pow(self.e)

    def inputs(self):
        return [
            torch.empty(5, 5),  # x
            torch.empty(5, 5),  # y
            torch.empty(5, 5),  # z
        ]

    def register(self, pm: PatternMatcherPass):
        register_replacement(self.pattern, self.replacement, self.inputs(), fwd_only, pm)

# This was my attempt to circumvent the issue, unsuccessful
class ReluSumPattern2(ReluSumPattern):
    def register(self, pm: PatternMatcherPass):
        # doesn't matter if using functools or not, doesn't work
        @functools.wraps(self.pattern)
        def wrapped_pattern(*args, **kwargs):
            self.pattern(*args, **kwargs)

        @functools.wraps(self.replacement)
        def wrapped_replacement(*args, **kwargs):
            self.pattern(*args, **kwargs)

        register_replacement(wrapped_pattern, wrapped_replacement, self.inputs(), fwd_only, pm)

# This works but it's a little clunkier, not bad for now though
class ReluSumPatternWorking(ReluSumPattern):
    def get_pattern_replacement(self):
        def pattern(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            return x.pow(self.e) + y.pow(self.e) + z.pow(self.e)
        def replacement(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            return (x + y + z).pow(self.e)

        return pattern, replacement

    def register(self, pm: PatternMatcherPass):
        pattern, replacement = self.get_pattern_replacement()
        register_replacement(pattern, replacement, self.inputs(), fwd_only, pm)


def empty_bf16(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.bfloat16)


def empty_fp8(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.float8_e4m3fn)


my_patterns = PatternMatcherPass()

ReluSumPatternWorking(2).register(my_patterns)
print(my_patterns.patterns)

# These don't work
try:
    ReluSumPattern2(3).register(my_patterns)
except Exception as e:
    print(e)
    traceback.print_exc()

print(my_patterns.patterns)

try:
    ReluSumPattern(4).register(my_patterns)
except Exception as e:
    print(e)
    traceback.print_exc()
print(my_patterns.patterns)

def custom_pass(graph: torch.fx.Graph) -> torch.fx.Graph:
    print(graph.python_code(root_module='self', include_stride=False).src)
    count = my_patterns.apply(graph)
    print(f"Count: {count}")
    graph.eliminate_dead_code()
    print(graph.python_code(root_module='self', include_stride=False).src)
    return graph


@torch.compile(options={'post_grad_custom_post_pass': custom_pass, 'enable_auto_functionalized_v2': False})
def my_func_static(x):
    y = x.relu()
    z = y.tanh()
    z2 = x.pow(2) + y.pow(2) + z.pow(2)
    z3 = x.pow(3) + y.pow(3) + z2.pow(3)
    z4 = x.pow(4) + y.pow(4) + z3.pow(4)
    return z4 + 5

print("Run my_func_static")
inputs = [torch.ones((5, 4))]
print(my_func_static(*inputs))