----------------------------- Captured stdout call -----------------------------
inline_call [("Unsupported function call
  Explanation: Dynamo does not know how to trace the function `<class 'collections.defaultdict'>`
  Hint: Avoid calling `<class 'collections.defaultdict'>` in your code.
  Hint: Please report an issue to PyTorch.

  Developer debug context:
call_function UserDefinedClassVariable(<class 'collections.defaultdict'>) [GetAttrVariable(DefaultDictVariable(), default_factory), ConstDictVariable()] {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0147.html", 1)]
- generated xml file: /var/lib/jenkins/workspace/test/test-reports/python-pytest/dynamo.test_misc/dynamo.test_misc-271de5e392c25fc0.xml -
=========================== short test summary info ============================
FAILED [0.2732s] dynamo/test_misc.py::MiscTestsPyTree::test_pytree_tree_map_dict_order_cxx - torch._dynamo.exc.Unsupported: Unsupported function call
  Explanation: Dynamo does not know how to trace the function `<class 'collections.defaultdict'>`
  Hint: Avoid calling `<class 'collections.defaultdict'>` in your code.
  Hint: Please report an issue to PyTorch.

  Developer debug context: call_function UserDefinedClassVariable(<class 'collections.defaultdict'>) [GetAttrVariable(DefaultDictVariable(), default_factory), ConstDictVariable()] {}