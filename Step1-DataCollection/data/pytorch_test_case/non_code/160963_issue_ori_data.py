pid = os.getpid()
wid = torch.utils.data.get_worker_info().id
print(f"[PID={pid}/WID={wid}]")