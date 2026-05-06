Traceback (most recent call last):
  File "/pytorch/test/test_type_hints.py", line 145, in test_doc_examples
    self.fail(f"mypy failed:\n{stderr}\n{stdout}")
  File "/usr/lib64/python3.11/unittest/case.py", line 703, in fail
    raise self.failureException(msg)
AssertionError: mypy failed:

test/generated_type_hints_smoketest.py:35:9: error: Name "ParamSpec" is not defined  [name-defined]
test/generated_type_hints_smoketest.py:36:12: error: Name "get_origin" is not defined  [name-defined]
test/generated_type_hints_smoketest.py:36:23: error: Name "Literal" is not defined  [name-defined]
test/generated_type_hints_smoketest.py:37:12: error: Name "get_origin" is not defined  [name-defined]
test/generated_type_hints_smoketest.py:38:12: error: Name "get_origin" is not defined  [name-defined]
test/generated_type_hints_smoketest.py:38:23: error: Name "ClassVar" is not defined  [name-defined]
test/generated_type_hints_smoketest.py:39:12: error: Name "get_origin" is not defined  [name-defined]
test/generated_type_hints_smoketest.py:39:23: error: Name "Generic" is not defined  [name-defined]
test/generated_type_hints_smoketest.py:40:12: error: Name "get_origin" is not defined  [name-defined]
test/generated_type_hints_smoketest.py:40:23: error: Name "Generic" is not defined  [name-defined]
test/generated_type_hints_smoketest.py:40:31: error: Name "T" is not defined  [name-defined]
test/generated_type_hints_smoketest.py:41:12: error: Name "get_origin" is not defined  [name-defined]
test/generated_type_hints_smoketest.py:41:23: error: Name "Union" is not defined  [name-defined]
test/generated_type_hints_smoketest.py:41:23: note: Did you forget to import it from "typing"? (Suggestion: "from typing import Union")
test/generated_type_hints_smoketest.py:41:29: error: Name "T" is not defined  [name-defined]
test/generated_type_hints_smoketest.py:42:12: error: Name "get_origin" is not defined  [name-defined]
test/generated_type_hints_smoketest.py:42:23: error: Name "List" is not defined  [name-defined]
test/generated_type_hints_smoketest.py:42:23: note: Did you forget to import it from "typing"? (Suggestion: "from typing import List")
test/generated_type_hints_smoketest.py:42:28: error: Name "Tuple" is not defined  [name-defined]
test/generated_type_hints_smoketest.py:42:28: note: Did you forget to import it from "typing"? (Suggestion: "from typing import Tuple")
test/generated_type_hints_smoketest.py:42:34: error: Name "T" is not defined  [name-defined]
test/generated_type_hints_smoketest.py:43:12: error: Name "get_origin" is not defined  [name-defined]
test/generated_type_hints_smoketest.py:1364:43: error: "Storage" has no attribute "cuda"  [attr-defined]
test/generated_type_hints_smoketest.py:2852:61: error: Argument "ambiguity_check" to "dim_order" of "Tensor" has incompatible type "str"; expected "bool | list[memory_format]"  [arg-type]
Found 21 errors in 1 file (checked 1 source file)