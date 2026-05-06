# insert after https://github.com/pytorch/pytorch/blob/main/torch/fx/experimental/symbolic_shapes.py#L603
            if isinstance(u1, float):
                log.info(
                    "rebind_unbacked: discard %s %s %s -> %s",
                    n.target,
                    raw_u0,
                    path,
                    u1,
                )
                continue