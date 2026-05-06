[2100/2183] Building CXX object functorch\CMakeFiles\functorch.dir\csrc\init_dim_only.cpp.obj
Traceback (most recent call last):
2025-09-18T13:34:51.9648235Z   File "<frozen runpy>", line 198, in _run_module_as_main
2025-09-18T13:34:51.9648935Z   File "<frozen runpy>", line 88, in _run_code
2025-09-18T13:34:51.9649374Z   File "C:\a\pytorch\pytorch\pytorch\.venv\Scripts\cmake.exe\__main__.py", line 6, in <module>
2025-09-18T13:34:51.9659496Z   File "C:\a\pytorch\pytorch\pytorch\.venv\Lib\site-packages\cmake\__init__.py", line 57, in cmake
2025-09-18T13:34:51.9675180Z     _program_exit('cmake', *sys.argv[1:])
2025-09-18T13:34:51.9677330Z   File "C:\a\pytorch\pytorch\pytorch\.venv\Lib\site-packages\cmake\__init__.py", line 47, in _program_exit
2025-09-18T13:34:51.9679816Z     raise SystemExit(_program(name, args))
2025-09-18T13:34:51.9680612Z                      ^^^^^^^^^^^^^^^^^^^^
2025-09-18T13:34:51.9681017Z   File "C:\a\pytorch\pytorch\pytorch\.venv\Lib\site-packages\cmake\__init__.py", line 43, in _program
2025-09-18T13:34:51.9682658Z     return subprocess.call([os.path.join(CMAKE_BIN_DIR, name), *args], close_fds=False)
2025-09-18T13:34:51.9684131Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-09-18T13:34:51.9684556Z   File "C:\temp\dependencies\Python\Lib\subprocess.py", line 391, in call
2025-09-18T13:34:51.9690137Z     return p.wait(timeout=timeout)
2025-09-18T13:34:51.9690397Z            ^^^^^^^^^^^^^^^^^^^^^^^
2025-09-18T13:34:51.9690766Z   File "C:\temp\dependencies\Python\Lib\subprocess.py", line 1264, in wait
2025-09-18T13:34:51.9695216Z     return self._wait(timeout=timeout)
2025-09-18T13:34:51.9695507Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-09-18T13:34:51.9697150Z   File "C:\temp\dependencies\Python\Lib\subprocess.py", line 1590, in _wait
2025-09-18T13:34:51.9700238Z     result = _winapi.WaitForSingleObject(self._handle,
2025-09-18T13:34:51.9700567Z              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-09-18T13:34:51.9700803Z KeyboardInterrupt
2025-09-18T13:34:52.0059091Z -- Checkout nccl release tag: v2.27.5-1
2025-09-18T13:34:52.0934290Z Terminate batch job (Y/N)? 
2025-09-18T13:34:59.4687376Z Terminate batch job (Y/N)? 
2025-09-18T13:34:59.4696642Z "Failed on build_pytorch. (exitcode = -1073741510)"
2025-09-18T13:34:59.4737167Z ^C
2025-09-18T13:34:59.4817414Z ##[error]The operation was canceled.