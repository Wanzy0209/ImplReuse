F:\syan\softwares\miniconda\miniconda3\envs\TeRL\python.exe F:\syan\TeRL整理\TeRL\Step5-Evaluate\data-analysis\api_coverage_analysis.py 
======================================================================
API Coverage Analysis
======================================================================

Processing cross_baseline (framework: tf)...
  Found 267 files
  Saved detailed coverage to F:\syan\TeRL整理\TeRL\Step5-Evaluate\data-analysis\cross_baseline_api_coverage.csv
  Summary:
    Total files: 267
    Total unique tf APIs: 360

Processing cross_reuse (framework: tf)...
  Found 2082 files
  Saved detailed coverage to F:\syan\TeRL整理\TeRL\Step5-Evaluate\data-analysis\cross_reuse_api_coverage.csv
  Summary:
    Total files: 2082
    Total unique tf APIs: 1329

Processing single_baseline (framework: torch)...
  Found 267 files
  Saved detailed coverage to F:\syan\TeRL整理\TeRL\Step5-Evaluate\data-analysis\single_baseline_api_coverage.csv
  Summary:
    Total files: 267
    Total unique torch APIs: 408

Processing single_reuse (framework: torch)...
  Found 1253 files
  Saved detailed coverage to F:\syan\TeRL整理\TeRL\Step5-Evaluate\data-analysis\single_reuse_api_coverage.csv
  Summary:
    Total files: 1253
    Total unique torch APIs: 676

Saved summary to F:\syan\TeRL整理\TeRL\Step5-Evaluate\data-analysis\api_coverage_summary.csv

======================================================================
Overall Summary
======================================================================

[Cross - TensorFlow]
  cross_baseline unique APIs: 360
  cross_reuse unique APIs: 1329
  Only in baseline: 114
  Only in reuse: 1083
  Intersection (both): 246
  Union (total): 1443
  Formula check: 114 + 1083 + 246 = 1443

[Single - PyTorch]
  single_baseline unique APIs: 408
  single_reuse unique APIs: 676
  Only in baseline: 183
  Only in reuse: 451
  Intersection (both): 225
  Union (total): 859
  Formula check: 183 + 451 + 225 = 859

Process finished with exit code 0
