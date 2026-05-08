shrink_group(ranks_to_exclude: List[int],
            Pg: Optional[ProcessGroup] = None,
            shrink_flags : int = NCCL_SHRINK_DEFAULT)

Shrink an existing distributed group. Only group members of the updated ProcessGroup need to enter this function. The excluded ranks do not need to call this function. The scope of this call is therefore collective across the processes that belong to the shrunk distributed group.

Args:
ranks_exclude (list[int]):  List of group ranks to be excluded in the updated ProcessGroup. This list must be consistent across all participating processes.
pg (ProcessGroup, optional): The process group to work on, If None, the default process group will be used.
shrink_flags (int): NCCL_SHRINK_DEFAULT (default)
                    NCCL_SHRINK_ABORT (attempt to terminate ongoing operations in the parent communicator before shrinking.