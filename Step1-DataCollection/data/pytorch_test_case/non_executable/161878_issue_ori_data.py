/workspace/pytorch# bash inductor_single_run.sh multiple inference performance torchbench BERT_pytorch amp first static cpp
Testing with cpp wrapper.
Testing with inductor.
multi-threads testing....
loading model: 0it [00:01, ?it/s]
cpu  eval  BERT_pytorch
skipping cudagraphs due to cpp wrapper enabled
running benchmark: 100%|███████████████████████████████████████████████████████████████████████████| 50/50 [00:08<00:00,  5.62it/s]
2.014x
WARNING:common:Trying to call the empty_gpu_cache for device: cpu, which is not in list [cuda, xpu]
dev,name,batch_size,speedup,abs_latency,compilation_latency,compression_ratio,eager_peak_mem,dynamo_peak_mem,calls_captured,unique_graphs,graph_breaks,unique_graph_breaks,autograd_captures,autograd_compiles,cudagraph_skips
cpu,BERT_pytorch,2,2.014039,58.901280,38.538333,0.923878,250.866893,271.536947,545,1,0,0,0,0,1