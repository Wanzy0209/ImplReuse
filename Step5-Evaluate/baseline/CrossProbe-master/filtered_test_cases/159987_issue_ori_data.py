import torch.distributed as dist

def test_new_subgroups_with_group_param():
    self._init_global_test()
    init_multigpu_helper(dist.get_world_size(), BACKEND)
    cur_subgroup, subgroups = dist.new_subgroups_by_enumeration(
        ranks_per_subgroup_list=[[0, 2], [1, 3]]
    )