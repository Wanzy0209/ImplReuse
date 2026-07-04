# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import copy
import math
import torch
import argparse
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)

torch.backends.cudnn.allow_tf32=False

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
args = parser.parse_args()

model_name = "bert-base-cased"
config = AutoConfig.from_pretrained(
    model_name,
    cache_dir=None,
)


model = AutoModelForQuestionAnswering.from_pretrained(
    model_name,
    from_tf=False,
    config=config,
    cache_dir = None,
)

tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        do_lower_case=False,
        cache_dir= None,
        use_fast=False
    )

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')

torch.manual_seed(0)

input_x_cpu = encoded_input["input_ids"].expand(args.batch_size, 13).contiguous()
input_y_cpu = encoded_input["attention_mask"].expand(args.batch_size, 13).contiguous()
input_z_cpu = encoded_input["token_type_ids"].expand(args.batch_size, 13).contiguous()
input_x_cuda = input_x_cpu.to("cuda")
input_y_cuda = input_y_cpu.to("cuda")
input_z_cuda = input_z_cpu.to("cuda")


# Load the pre-trained resnet50 model with ImageNet weights

model=model.eval()

with torch.inference_mode() and torch.amp.autocast("cuda"):
    device = 'cuda'
    aot_model = model.to(device=device)

    tensor_x = torch.randint(low=0, high=10, size=(args.batch_size, 384), dtype=torch.int64, device=device)
    tensor_y = torch.randint(low=0, high=10, size=(args.batch_size, 384), dtype=torch.int64, device=device)
    tensor_z = torch.randint(low=0, high=10, size=(args.batch_size, 384), dtype=torch.int64, device=device)
    aot_args = (tensor_x, tensor_y, tensor_z)
    aot_kwargs = {"return_dict":False}

    # batch_dim = torch.export.Dim("batch", min=1, max=1024)
    # sc = torch.export.ShapesCollection()
    # sc[tensor_x] = (batch_dim, 384)
    # sc[tensor_y] = (batch_dim, 384)
    # sc[tensor_z] = (batch_dim, 384)

    # exported = torch.export.export(aot_model, aot_args, aot_kwargs, dynamic_shapes=sc)
    exported = torch.export.export(aot_model, aot_args, aot_kwargs)
    output_path = torch._inductor.aoti_compile_and_package(exported, package_path=f"./bert_base_{device}.pt2")
    print(f"aot model output_path is {output_path}")

aot_model_cuda = torch._inductor.aoti_load_package("./bert_base_cuda.pt2")

with torch.no_grad() and torch.amp.autocast("cuda"):
    output_cuda = aot_model_cuda(input_x_cuda, input_y_cuda, input_z_cuda, return_dict=False)