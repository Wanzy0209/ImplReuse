============================================= test session starts ==============================================
platform linux -- Python 3.10.12, pytest-8.4.1, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /transformers
configfile: pyproject.toml
plugins: typeguard-4.4.4, hydra-core-1.3.2, anyio-4.10.0, hypothesis-6.138.3, asyncio-1.1.0, order-1.3.0, rerunfailures-15.1, rich-0.2.0, timeout-2.4.0, xdist-3.8.0
asyncio: mode=strict, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function
collecting ... Fatal Python error: Aborted

Thread 0x00007f5408683640 (most recent call first):
  File "/usr/lib/python3.10/socket.py", line 293 in accept
  File "/usr/local/lib/python3.10/dist-packages/pytest_rerunfailures.py", line 438 in run_server
  File "/usr/lib/python3.10/threading.py", line 953 in run
  File "/usr/lib/python3.10/threading.py", line 1016 in _bootstrap_inner
  File "/usr/lib/python3.10/threading.py", line 973 in _bootstrap

Current thread 0x00007f55c68fa000 (most recent call first):
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 1176 in create_module
  File "<frozen importlib._bootstrap>", line 571 in module_from_spec
  File "<frozen importlib._bootstrap>", line 674 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap>", line 1078 in _handle_fromlist
  File "/usr/local/lib/python3.10/dist-packages/detectron2/layers/deform_conv.py", line 506 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "/usr/local/lib/python3.10/dist-packages/detectron2/layers/__init__.py", line 3 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "/usr/local/lib/python3.10/dist-packages/detectron2/modeling/__init__.py", line 2 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "/transformers/src/transformers/models/layoutlmv2/modeling_layoutlmv2.py", line 43 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "<frozen importlib._bootstrap>", line 1050 in _gcd_import
  File "/usr/lib/python3.10/importlib/__init__.py", line 126 in import_module
  File "/transformers/src/transformers/utils/import_utils.py", line 2327 in _get_module
  File "/transformers/src/transformers/utils/import_utils.py", line 2299 in __getattr__
  File "/transformers/src/transformers/models/auto/auto_factory.py", line 729 in getattribute_from_module
  File "/transformers/src/transformers/models/auto/auto_factory.py", line 817 in _load_attr_from_module
  File "/transformers/src/transformers/models/auto/auto_factory.py", line 848 in <listcomp>
  File "/transformers/src/transformers/models/auto/auto_factory.py", line 845 in items
  File "/transformers/tests/pipelines/test_pipelines_question_answering.py", line 54 in QAPipelineTests
  File "/transformers/tests/pipelines/test_pipelines_question_answering.py", line 49 in <module>
  File "/usr/local/lib/python3.10/dist-packages/_pytest/assertion/rewrite.py", line 186 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "/transformers/tests/test_pipeline_mixin.py", line 78 in <module>
  File "/usr/local/lib/python3.10/dist-packages/_pytest/assertion/rewrite.py", line 186 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "/transformers/tests/causal_lm_tester.py", line 38 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "/transformers/tests/models/gemma/test_modeling_gemma.py", line 38 in <module>
  File "/usr/local/lib/python3.10/dist-packages/_pytest/assertion/rewrite.py", line 186 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "<frozen importlib._bootstrap>", line 1050 in _gcd_import
  File "/usr/lib/python3.10/importlib/__init__.py", line 126 in import_module
  File "/usr/local/lib/python3.10/dist-packages/_pytest/pathlib.py", line 587 in import_path
  File "/usr/local/lib/python3.10/dist-packages/_pytest/python.py", line 498 in importtestmodule
  File "/usr/local/lib/python3.10/dist-packages/_pytest/python.py", line 551 in _getobj
  File "/usr/local/lib/python3.10/dist-packages/_pytest/python.py", line 280 in obj
  File "/usr/local/lib/python3.10/dist-packages/_pytest/python.py", line 567 in _register_setup_module_fixture
  File "/usr/local/lib/python3.10/dist-packages/_pytest/python.py", line 554 in collect
  File "/usr/local/lib/python3.10/dist-packages/_pytest/runner.py", line 389 in collect
  File "/usr/local/lib/python3.10/dist-packages/_pytest/runner.py", line 344 in from_call
  File "/usr/local/lib/python3.10/dist-packages/_pytest/runner.py", line 391 in pytest_make_collect_report
  File "/usr/local/lib/python3.10/dist-packages/pluggy/_callers.py", line 121 in _multicall
  File "/usr/local/lib/python3.10/dist-packages/pluggy/_manager.py", line 120 in _hookexec
  File "/usr/local/lib/python3.10/dist-packages/pluggy/_hooks.py", line 512 in __call__
  File "/usr/local/lib/python3.10/dist-packages/_pytest/runner.py", line 567 in collect_one_node
  File "/usr/local/lib/python3.10/dist-packages/_pytest/main.py", line 839 in _collect_one_node
  File "/usr/local/lib/python3.10/dist-packages/_pytest/main.py", line 916 in collect
  File "/usr/local/lib/python3.10/dist-packages/_pytest/runner.py", line 389 in collect
  File "/usr/local/lib/python3.10/dist-packages/_pytest/runner.py", line 344 in from_call
  File "/usr/local/lib/python3.10/dist-packages/_pytest/runner.py", line 391 in pytest_make_collect_report
  File "/usr/local/lib/python3.10/dist-packages/pluggy/_callers.py", line 121 in _multicall
  File "/usr/local/lib/python3.10/dist-packages/pluggy/_manager.py", line 120 in _hookexec
  File "/usr/local/lib/python3.10/dist-packages/pluggy/_hooks.py", line 512 in __call__
  File "/usr/local/lib/python3.10/dist-packages/_pytest/runner.py", line 567 in collect_one_node
  File "/usr/local/lib/python3.10/dist-packages/_pytest/main.py", line 793 in perform_collect
  File "/usr/local/lib/python3.10/dist-packages/_pytest/main.py", line 353 in pytest_collection
  File "/usr/local/lib/python3.10/dist-packages/pluggy/_callers.py", line 121 in _multicall
  File "/usr/local/lib/python3.10/dist-packages/pluggy/_manager.py", line 120 in _hookexec
  File "/usr/local/lib/python3.10/dist-packages/pluggy/_hooks.py", line 512 in __call__
  File "/usr/local/lib/python3.10/dist-packages/_pytest/main.py", line 342 in _main
  File "/usr/local/lib/python3.10/dist-packages/_pytest/main.py", line 289 in wrap_session
  File "/usr/local/lib/python3.10/dist-packages/_pytest/main.py", line 336 in pytest_cmdline_main
  File "/usr/local/lib/python3.10/dist-packages/pluggy/_callers.py", line 121 in _multicall
  File "/usr/local/lib/python3.10/dist-packages/pluggy/_manager.py", line 120 in _hookexec
  File "/usr/local/lib/python3.10/dist-packages/pluggy/_hooks.py", line 512 in __call__
  File "/usr/local/lib/python3.10/dist-packages/_pytest/config/__init__.py", line 175 in main
  File "/usr/local/lib/python3.10/dist-packages/_pytest/config/__init__.py", line 201 in console_main
  File "/usr/local/lib/python3.10/dist-packages/pytest/__main__.py", line 9 in <module>
  ...

Extension modules: yaml._yaml, charset_normalizer.md, regex._regex, numpy.core._multiarray_umath, numpy.core._mu
ltiarray_tests, numpy.linalg._umath_linalg, numpy.fft._pocketfft_internal, numpy.random._common, numpy.random.bi
t_generator, numpy.random._bounded_integers, numpy.random._mt19937, numpy.random.mtrand, numpy.random._philox, n
umpy.random._pcg64, numpy.random._sfc64, numpy.random._generator, torch._C, torch._C._dynamo.autograd_compiler,
torch._C._dynamo.eval_frame, torch._C._dynamo.guards, torch._C._dynamo.utils, torch._C._fft, torch._C._linalg, t
orch._C._nested, torch._C._nn, torch._C._sparse, torch._C._special, markupsafe._speedups, PIL._imaging, scipy._l
ib._ccallback_c, scipy.linalg._fblas, scipy.linalg._flapack, scipy.linalg.cython_lapack, scipy.linalg._cythonize
d_array_utils, scipy.linalg._solve_toeplitz, scipy.linalg._flinalg, scipy.linalg._decomp_lu_cython, scipy.linalg
._matfuncs_sqrtm_triu, scipy.linalg.cython_blas, scipy.linalg._matfuncs_expm, scipy.linalg._decomp_update, scipy
.sparse._sparsetools, _csparsetools, scipy.sparse._csparsetools, scipy.sparse.linalg._dsolve._superlu, scipy.spa
rse.linalg._eigen.arpack._arpack, scipy.sparse.csgraph._tools, scipy.sparse.csgraph._shortest_path, scipy.sparse
.csgraph._traversal, scipy.sparse.csgraph._min_spanning_tree, scipy.sparse.csgraph._flow, scipy.sparse.csgraph._
matching, scipy.sparse.csgraph._reordering, scipy.optimize._minpack2, scipy.optimize._group_columns, scipy._lib.
messagestream, scipy.optimize._trlib._trlib, scipy.optimize._lbfgsb, _moduleTNC, scipy.optimize._moduleTNC, scip
y.optimize._cobyla, scipy.optimize._slsqp, scipy.optimize._minpack, scipy.optimize._lsq.givens_elimination, scip
y.optimize._zeros, scipy.optimize._highs.cython.src._highs_wrapper, scipy.optimize._highs._highs_wrapper, scipy.
optimize._highs.cython.src._highs_constants, scipy.optimize._highs._highs_constants, scipy.linalg._interpolative
, scipy.optimize._bglu_dense, scipy.optimize._lsap, scipy.spatial._ckdtree, scipy.spatial._qhull, scipy.spatial.
_voronoi, scipy.spatial._distance_wrap, scipy.spatial._hausdorff, scipy.special._ufuncs_cxx, scipy.special._ufun
cs, scipy.special._specfun, scipy.special._comb, scipy.special._ellip_harm_2, scipy.spatial.transform._rotation,
 scipy.optimize._direct, PIL._imagingft, av._core, av.logging, av.bytesource, av.buffer, av.audio.format, av.err
or, av.dictionary, av.container.pyio, av.utils, av.option, av.descriptor, av.format, av.stream, av.container.str
eams, av.sidedata.motionvectors, av.sidedata.sidedata, av.opaque, av.packet, av.container.input, av.container.ou
tput, av.container.core, av.codec.context, av.video.format, av.video.reformatter, av.plane, av.video.plane, av.v
ideo.frame, av.video.stream, av.codec.hwaccel, av.codec.codec, av.frame, av.audio.layout, av.audio.plane, av.aud
io.frame, av.audio.stream, av.filter.pad, av.filter.link, av.filter.context, av.filter.graph, av.filter.filter,
av.filter.loudnorm, av.audio.resampler, av.audio.codeccontext, av.audio.fifo, av.bitstream, av.video.codeccontex
t, sklearn.__check_build._check_build, psutil._psutil_linux, psutil._psutil_posix, scipy.ndimage._nd_image, _ni_
label, scipy.ndimage._ni_label, scipy.integrate._odepack, scipy.integrate._quadpack, scipy.integrate._vode, scip
y.integrate._dop, scipy.integrate._lsoda, scipy.special.cython_special, scipy.stats._stats, scipy.stats.beta_ufu
nc, scipy.stats._boost.beta_ufunc, scipy.stats.binom_ufunc, scipy.stats._boost.binom_ufunc, scipy.stats.nbinom_u
func, scipy.stats._boost.nbinom_ufunc, scipy.stats.hypergeom_ufunc, scipy.stats._boost.hypergeom_ufunc, scipy.st
ats.ncf_ufunc, scipy.stats._boost.ncf_ufunc, scipy.stats.ncx2_ufunc, scipy.stats._boost.ncx2_ufunc, scipy.stats.
nct_ufunc, scipy.stats._boost.nct_ufunc, scipy.stats.skewnorm_ufunc, scipy.stats._boost.skewnorm_ufunc, scipy.st
ats.invgauss_ufunc, scipy.stats._boost.invgauss_ufunc, scipy.interpolate._fitpack, scipy.interpolate.dfitpack, s
cipy.interpolate._bspl, scipy.interpolate._ppoly, scipy.interpolate.interpnd, scipy.interpolate._rbfinterp_pythr
an, scipy.interpolate._rgi_cython, scipy.stats._biasedurn, scipy.stats._levy_stable.levyst, scipy.stats._stats_p
ythran, scipy._lib._uarray._uarray, scipy.stats._ansari_swilk_statistics, scipy.stats._sobol, scipy.stats._qmc_c
y, scipy.stats._mvn, scipy.stats._rcont.rcont, scipy.stats._unuran.unuran_wrapper, pyarrow.lib, pandas._libs.tsl
ibs.ccalendar, pandas._libs.tslibs.np_datetime, pandas._libs.tslibs.dtypes, pandas._libs.tslibs.base, pandas._li
bs.tslibs.nattype, pandas._libs.tslibs.timezones, pandas._libs.tslibs.fields, pandas._libs.tslibs.timedeltas, pa
ndas._libs.tslibs.tzconversion, pandas._libs.tslibs.timestamps, pandas._libs.properties, pandas._libs.tslibs.off
sets, pandas._libs.tslibs.strptime, pandas._libs.tslibs.parsing, pandas._libs.tslibs.conversion, pandas._libs.ts
libs.period, pandas._libs.tslibs.vectorized, pandas._libs.ops_dispatch, pandas._libs.missing, pandas._libs.hasht
able, pandas._libs.algos, pandas._libs.interval, pandas._libs.lib, pyarrow._compute, pandas._libs.ops, pandas._l
ibs.hashing, pandas._libs.arrays, pandas._libs.tslib, pandas._libs.sparse, pandas._libs.internals, pandas._libs.
indexing, pandas._libs.index, pandas._libs.writers, pandas._libs.join, pandas._libs.window.aggregations, pandas.
_libs.window.indexers, pandas._libs.reshape, pandas._libs.groupby, pandas._libs.json, pandas._libs.parsers, pand
as._libs.testing, _cyutility, sklearn._cyutility, sklearn.utils._isfinite, sklearn.utils.sparsefuncs_fast, sklea
rn.utils.murmurhash, sklearn.utils._openmp_helpers, sklearn.metrics.cluster._expected_mutual_info_fast, sklearn.
preprocessing._csr_polynomial_expansion, sklearn.preprocessing._target_encoder_fast, sklearn.metrics._dist_metri
cs, sklearn.metrics._pairwise_distances_reduction._datasets_pair, sklearn.utils._cython_blas, sklearn.metrics._p
airwise_distances_reduction._base, sklearn.metrics._pairwise_distances_reduction._middle_term_computer, sklearn.
utils._heap, sklearn.utils._sorting, sklearn.metrics._pairwise_distances_reduction._argkmin, sklearn.metrics._pa
irwise_distances_reduction._argkmin_classmode, sklearn.utils._vector_sentinel, sklearn.metrics._pairwise_distanc
es_reduction._radius_neighbors, sklearn.metrics._pairwise_distances_reduction._radius_neighbors_classmode, sklea
rn.metrics._pairwise_fast, _cffi_backend, pyarrow._parquet, pyarrow._fs, pyarrow._azurefs, pyarrow._hdfs, pyarro
w._gcsfs, pyarrow._s3fs, multidict._multidict, yarl._quoting_c, propcache._helpers_c, aiohttp._http_writer, aioh
ttp._http_parser, aiohttp._websocket.mask, aiohttp._websocket.reader_c, frozenlist._frozenlist, xxhash._xxhash,
pyarrow._acero, pyarrow._csv, pyarrow._json, pyarrow._substrait, pyarrow._dataset, pyarrow._dataset_orc, pyarrow
._parquet_encryption, pyarrow._dataset_parquet_encryption, pyarrow._dataset_parquet, PIL._imagingmath (total: 269)
Aborted (core dumped)