device = torch.device("cuda")
example_input = torch.randn((64, 3, 64, 64)).to(device)

# Trace the model
traced_model = torch.jit.trace(model, example_input)
traced_model = torch.jit.optimize_for_inference(traced_model)
traced_model.save(os.path.join(repo_dir, "opt_traced_model.pt"))