F:\syan\softwares\miniconda\miniconda3\envs\TeRL\python.exe F:\syan\TeRL整理\TeRL\Step5-Evaluate\data-analysis\categorize_results.py 
======================================================================
Categorizing Test Results
======================================================================

=== Processing cross_baseline_results ===
Processing 267 files (0 .log + 267 .json) in F:\syan\TeRL整理\TeRL\Step5-Evaluate\data-analysis\test_results\cross_baseline_results...
Saved categorized results to F:\syan\TeRL整理\TeRL\Step5-Evaluate\data-analysis\cross_baseline_results_categorized_results.csv

cross_baseline_results Category Summary:
  Success: 73
  Code Generation Error: 3
  API Misuse: 125
  Missing Dependency: 34
  Environment Failure: 3
  Execution Failure: 2
  Unknown Failure: 25
  Assertion Mismatch: 2

=== Processing cross_results ===
Processing 2082 files (2082 .log + 0 .json) in F:\syan\TeRL整理\TeRL\Step5-Evaluate\data-analysis\test_results\cross_results...
Saved categorized results to F:\syan\TeRL整理\TeRL\Step5-Evaluate\data-analysis\cross_results_categorized_results.csv

cross_results Category Summary:
  Success: 1275
  Missing Dependency: 296
  Assertion Mismatch: 118
  API Misuse: 277
  Unknown Failure: 48
  Environment Failure: 42
  Execution Failure: 20
  Timeout: 5
  Code Generation Error: 1

=== Processing single_baseline_results ===
Processing 267 files (0 .log + 267 .json) in F:\syan\TeRL整理\TeRL\Step5-Evaluate\data-analysis\test_results\single_baseline_results...
Saved categorized results to F:\syan\TeRL整理\TeRL\Step5-Evaluate\data-analysis\single_baseline_results_categorized_results.csv

single_baseline_results Category Summary:
  Unknown Failure: 68
  Code Generation Error: 5
  API Misuse: 97
  Missing Dependency: 52
  Success: 34
  Assertion Mismatch: 2
  Environment Failure: 9

=== Processing single_results ===
Processing 1253 files (1253 .log + 0 .json) in F:\syan\TeRL整理\TeRL\Step5-Evaluate\data-analysis\test_results\single_results...
Saved categorized results to F:\syan\TeRL整理\TeRL\Step5-Evaluate\data-analysis\single_results_categorized_results.csv

single_results Category Summary:
  Success: 877
  API Misuse: 247
  Unknown Failure: 65
  Missing Dependency: 17
  Environment Failure: 20
  Assertion Mismatch: 17
  Timeout: 3
  Execution Failure: 7

======================================================================
Overall Summary
======================================================================

cross_baseline_results:
  Success: 73 (27.3%)
  Code Generation Error: 3 (1.1%)
  API Misuse: 125 (46.8%)
  Missing Dependency: 34 (12.7%)
  Environment Failure: 3 (1.1%)
  Execution Failure: 2 (0.7%)
  Unknown Failure: 25 (9.4%)
  Assertion Mismatch: 2 (0.7%)
  Success Rate: 73/267 (27.3%)

cross_results:
  Success: 1275 (61.2%)
  Missing Dependency: 296 (14.2%)
  Assertion Mismatch: 118 (5.7%)
  API Misuse: 277 (13.3%)
  Unknown Failure: 48 (2.3%)
  Environment Failure: 42 (2.0%)
  Execution Failure: 20 (1.0%)
  Timeout: 5 (0.2%)
  Code Generation Error: 1 (0.0%)
  Success Rate: 1275/2082 (61.2%)

single_baseline_results:
  Unknown Failure: 68 (25.5%)
  Code Generation Error: 5 (1.9%)
  API Misuse: 97 (36.3%)
  Missing Dependency: 52 (19.5%)
  Success: 34 (12.7%)
  Assertion Mismatch: 2 (0.7%)
  Environment Failure: 9 (3.4%)
  Success Rate: 34/267 (12.7%)

single_results:
  Success: 877 (70.0%)
  API Misuse: 247 (19.7%)
  Unknown Failure: 65 (5.2%)
  Missing Dependency: 17 (1.4%)
  Environment Failure: 20 (1.6%)
  Assertion Mismatch: 17 (1.4%)
  Timeout: 3 (0.2%)
  Execution Failure: 7 (0.6%)
  Success Rate: 877/1253 (70.0%)

======================================================================
Combined Summary
======================================================================
  Success: 2259 (58.4%)
  Code Generation Error: 9 (0.2%)
  API Misuse: 746 (19.3%)
  Missing Dependency: 399 (10.3%)
  Environment Failure: 74 (1.9%)
  Execution Failure: 29 (0.7%)
  Unknown Failure: 206 (5.3%)
  Assertion Mismatch: 139 (3.6%)
  Timeout: 8 (0.2%)
  Overall Success Rate: 2259/3869 (58.4%)

Process finished with exit code 0
