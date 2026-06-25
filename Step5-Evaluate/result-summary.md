读取到 267 个 issue_id

处理完成，共 267 个有效数据

统计信息:

跨框架复用测试用例总数: 2082

单框架复用测试用例总数: 1253

复用测试用例总数: 3335

跨库复用测试用例执行情况：

Cross-Framework:
  Success: 1275（61.24%）
  Missing Dependency: 296
  Assertion Mismatch: 118
  API Misuse: 277
  Unknown Failure: 48
  Environment Failure: 42
  Execution Failure: 20
  Timeout: 5
  Code Generation Error: 1

Single-Framework:
  Success: 877（69.99%）
  API Misuse: 247
  Unknown Failure: 65
  Missing Dependency: 17
  Environment Failure: 20
  Assertion Mismatch: 17
  Timeout: 3
  Execution Failure: 7

（没执行完）Cross-Framework: 257 success, 95 fail

更正：Cross-Framework: 257 - 15 = 242 success, 95 + 15 = 110 fail

出错：3

可执行率：242 / (242 + 110) = 68.75%
