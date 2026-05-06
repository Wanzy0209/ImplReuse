from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.tensor import distribute_tensor, Shard
from torch.distributed.device_mesh import init_device_mesh
import torch
import torch.distributed as dist
import time

pg = torch.distributed.init_process_group(backend="gloo")
world_size = dist.get_world_size(pg)
rank = dist.get_rank(pg)
device_mesh = init_device_mesh("cpu", (world_size,))

fully_tensor = [torch.ones(1024, 1) for _ in range(1024)]
sharded_tensor = [distribute_tensor(tensor=t, device_mesh=device_mesh, placements=[Shard(0)]) for t in fully_tensor]
state_dict = {str(key): value for key, value in enumerate(sharded_tensor)}

planner = DefaultSavePlanner()
planner.set_up_planner(state_dict=state_dict, is_coordinator=rank==0)
local_plan = planner.create_local_plan()
gather_objs = [None] * world_size

dist.gather_object(obj=local_plan, object_gather_list=gather_objs if rank == 0 else None, dst=0, group=pg)

if rank == 0:
    start = time.time()
    all_local_plans, global_metadata = planner.create_global_plan(gather_objs)
    end = time.time()
    print(f"Create global planner cost {end - start}s")

dist.barrier()