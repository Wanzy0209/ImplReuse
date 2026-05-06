++ pip install -qr requirements.txt
  error: subprocess-exited-with-error
  
  × Preparing metadata (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [9 lines of output]
      💥 maturin failed
        Caused by: Cargo metadata failed. Does your crate compile with `cargo build`?
        Caused by: `cargo metadata` exited with an error: error: failed to parse lock file at: 
      
      Caused by:
        lock file version `4` was found, but this version of Cargo does not understand this lock file, perhaps Cargo needs to be updated?
      Error running maturin: Command '['maturin', 'pep517', 'write-dist-info', '--metadata-directory', '/tmp/pip-modern-metadata-z9pq1uvc', '--interpreter', '/opt/_internal/cpython-3.10.1/bin/python']' returned non-zero exit status 1.
      Checking for Rust toolchain....
      Running `maturin pep517 write-dist-info --metadata-directory /tmp/pip-modern-metadata-t3pm1ora --interpreter /opt/_internal/cpython-3.10.1/bin/python`
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.