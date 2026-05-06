wheelRuntimeError: Command bash /Users/ec2-user/runner/_work/_temp/exec_script failed with exit code 1

PackagesNotFoundError: The following packages are not available from current channels:

  - python=3.14

Current channels:

  - defaults

To search for alternate channels that may provide the conda package you're
looking for, navigate to

    https://anaconda.org/

and use the search bar at the top of the page.


Traceback (most recent call last):
  File "/Users/ec2-user/runner/_work/test-infra/test-infra/test-infra/.github/scripts/run_with_env_secrets.py", line 102, in <module>
    main()
    ~~~~^^
  File "/Users/ec2-user/runner/_work/test-infra/test-infra/test-infra/.github/scripts/run_with_env_secrets.py", line 61, in main
    run_cmd_or_die(f"bash { os.environ.get('RUNNER_TEMP', '') }/exec_script")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/ec2-user/runner/_work/test-infra/test-infra/test-infra/.github/scripts/run_with_env_secrets.py", line 39, in run_cmd_or_die
    raise RuntimeError(f"Command {cmd} failed with exit code {exit_code}")
RuntimeError: Command bash /Users/ec2-user/runner/_work/_temp/exec_script failed with exit code 1