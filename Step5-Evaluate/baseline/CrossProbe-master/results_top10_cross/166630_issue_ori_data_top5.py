```python
import tensorflow as tf

# Conversion: Replaced standard logger with tf.get_logger() for TensorFlow compatibility
# Conversion: Merged split string literals into a single format string to fix logging arguments
# Conversion: Added 'signals_to_handle' to the format string (present in dict but missing in original)
# Conversion: Renamed 'elastic_operator' to 'tensorflow_operator' to reflect the target framework
tf.get_logger().info(
    "Starting tensorflow_operator with launch configs:\n"
    "  entrypoint               : %(entrypoint)s\n"
    "  min_nodes                : %(min_nodes)s\n"
    "  max_nodes                : %(max_nodes)s\n"
    "  nproc_per_node           : %(nproc_per_node)s\n"
    "  run_id                   : %(run_id)s\n"
    "  rdzv_backend             : %(rdzv_backend)s\n"  # Maps to TF_CONFIG/ClusterSpec in TF
    "  rdzv_endpoint            : %(rdzv_endpoint)s\n"  # Maps to cluster address in TF
    "  rdzv_configs             : %(rdzv_configs)s\n"
    "  max_restarts             : %(max_restarts)s\n"
    "  monitor_interval         : %(monitor_interval)s\n"
    "  log_dir                  : %(log_dir)s\n"
    "  metrics_cfg              : %(metrics_cfg)s\n"
    "  event_log_handler        : %(event_log_handler)s\n"
    "  numa_options             : %(numa_options)s\n"
    "  signals_to_handle        : %(signals_to_handle)s\n"
    "  duplicate_stdout_filters : %(duplicate_stdout_filters)s\n"
    "  duplicate_stderr_filters : %(duplicate_stderr_filters)s\n"
    % {
        "entrypoint": entrypoint_name,
        "min_nodes": config.min_nodes,
        "max_nodes": config.max_nodes,
        "nproc_per_node": config.nproc_per_node,
        "run_id": config.run_id,
        "rdzv_backend": config.rdzv_backend,
        "rdzv_endpoint": config.rdzv_endpoint,
        "rdzv_configs": config.rdzv_configs,
        "max_restarts": config.max_restarts,
        "monitor_interval": config.monitor_interval,
        "log_dir": config.logs_specs.root_log_dir,  # type: ignore[union-attr]
        "metrics_cfg": config.metrics_cfg,
        "event_log_handler": config.event_log_handler,
        "numa_options": config.numa_options,
        "signals_to_handle": config.signals_to_handle,
        "duplicate_stdout_filters": config.duplicate_stdout_filters,
        "duplicate_stderr_filters": config.duplicate_stderr_filters,
    },
)
```