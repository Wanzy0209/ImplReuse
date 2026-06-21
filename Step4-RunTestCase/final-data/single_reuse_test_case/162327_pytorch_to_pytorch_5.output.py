import torch

print(torch.__version__, flush=True)

# Adapted test case for torch.nn.functional.instance_norm
# based on the heap-buffer-overflow inputs from max_unpool1d
input = [
    [
        torch.empty((5, 7, 4, 3, 7, 6), dtype=torch.int8), # input
        torch.empty((4, 9, 2), dtype=torch.int32),       # running_mean
        (),                                               # running_var
        None                                              # weight (Fixed: changed False to None)
    ],
    {},
    [],
    {}
]

# Attempt to trigger similar issues with instance_norm
torch.nn.functional.instance_norm(*input[0], **input[1])