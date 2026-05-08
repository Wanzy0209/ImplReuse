if config.joint_custom_pre_pass is not None:
        GraphTransformObserver(graph, "joint_custom_pre_pass").apply_graph_pass(
            config.joint_custom_pre_pass
        )
        count += 1

from .post_grad import remove_noop_ops

GraphTransformObserver(graph, "remove_noop_ops").apply_graph_pass(remove_noop_ops)

if config.joint_graph_constant_folding:
    GraphTransformObserver(graph, "constant_fold_uniform_value").apply_gm_pass(
        constant_fold_uniform_value
    )

if config.joint_custom_pre_pass is not None:
    GraphTransformObserver(graph, "joint_custom_pre_pass").apply_graph_pass(
        config.joint_custom_pre_pass
    )
    count += 1