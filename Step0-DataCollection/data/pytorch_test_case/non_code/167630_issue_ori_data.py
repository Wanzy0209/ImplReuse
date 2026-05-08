import contextlib
import gc
import logging
import os
import tempfile
from pathlib import Path

import torch
import torch._inductor
import torch.nn as nn

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
)


def log_current_memory() -> None:
    total = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    msg = "Current CUDA memory usage:"
    msg += f"\n  Total: {total / 1e9:.2f} GB"
    msg += f"\n  Allocated: {allocated / 1e9:.4} GB"
    msg += f"\n  Reserved: {reserved / 1e9:.4f} GB"
    logging.info(msg)


# ---------- toy model ----------
def make_mlp(in_dim=128, hidden=256, out_dim=64, depth=3):
    layers = []
    d = in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


def one_iter(i, device, batch, in_dim, hidden, out_dim, depth, workdir):
    model = make_mlp(in_dim, hidden, out_dim, depth).to(device).eval()
    x = torch.randn(batch, in_dim, device=device)
    with torch.inference_mode():
        _ = model(x)
        exported = torch.export.export(
            model,
            (x,),
        )

        pkg_path = Path(workdir) / f"mlp_{i}.pt2"
        path = torch._inductor.aoti_compile_and_package(  # returns artifact path
            exported_program=exported,
            package_path=str(pkg_path),
        )

    logging.info(f"[iter {i}] AOTI artifact: {path}")

    log_current_memory()

    del _
    del model, x, exported
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    with contextlib.suppress(OSError):
        os.remove(path)


def main():
    assert torch.cuda.is_available(), "CUDA is required for this MRE."

    device = "cuda"
    logging.info(f"Running on {torch.cuda.get_device_name(0)}")

    log_current_memory()
    for i in range(10):
        with tempfile.TemporaryDirectory() as tmp_workdir:
            one_iter(
                i=i,
                device=device,
                batch=32,
                in_dim=2048,
                hidden=512,
                out_dim=10,
                depth=6,
                workdir=tmp_workdir,
            )
    logging.info("Done.")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    log_current_memory()


if __name__ == "__main__":
    main()