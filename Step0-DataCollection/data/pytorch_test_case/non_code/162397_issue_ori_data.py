Traceback (most recent call last):
  File "/parse.py", line 19, in <module>
    main()
  File "/parse.py", line 11, in main
    parse_native_yaml(filename, "/pytorch/aten/src/ATen/native/tags.yaml")
  File "/usr/local/lib/python3.9/dist-packages/torchgen/gen.py", line 263, in parse_native_yaml
    _GLOBAL_PARSE_NATIVE_YAML_CACHE[path] = parse_native_yaml_struct(
  File "/usr/local/lib/python3.9/dist-packages/torchgen/gen.py", line 185, in parse_native_yaml_struct
    func, m = NativeFunction.from_yaml(e, loc, valid_tags, ignore_keys)
  File "/usr/local/lib/python3.9/dist-packages/torchgen/model.py", line 729, in from_yaml
    if t in valid_tags:
TypeError: unhashable type: 'dict'
  in crash-59ca5fa612a7a17ede0f2fb31c9a734d081e2911:495:
    acos(Tensor self) -> Tensor