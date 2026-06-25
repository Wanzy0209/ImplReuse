import torch
import os
os.environ['KINETO_USE_DAEMON'] = '1'
torch.utils.collect_env.main()