$ python3 scratch/mre_keras_no_sw.py
Keras: 3.11.3
PyTorch: 2.9.0
MPS available: True
X: (32, 50, 10) y: (32, 50, 1)
(mpsFileLoc): /AppleInternal/Library/BuildRoots/4~B5vFugC00QBfsSa38UwjlpDhP2WeLxB3cTjDRaE/Library/Caches/com.apple.xbs/Sources/MetalPerformanceShadersGraph/mpsgraph/MetalPerformanceShadersGraph/Core/Files/MPSGraphUtilities.mm:233:0: error: 'mps.multiply' op operands don't have broadcast-compatible shapes
(mpsFileLoc): /AppleInternal/Library/BuildRoots/4~B5vFugC00QBfsSa38UwjlpDhP2WeLxB3cTjDRaE/Library/Caches/com.apple.xbs/Sources/MetalPerformanceShadersGraph/mpsgraph/MetalPerformanceShadersGraph/Core/Files/MPSGraphUtilities.mm:233:0: note: see current operation: %10 = "mps.multiply"(%arg2, %9) : (tensor<32x50x1xf32>, tensor<32x50xf32>) -> tensor<*xf32>
(mpsFileLoc): /AppleInternal/Library/BuildRoots/4~B5vFugC00QBfsSa38UwjlpDhP2WeLxB3cTjDRaE/Library/Caches/com.apple.xbs/Sources/MetalPerformanceShadersGraph/mpsgraph/MetalPerformanceShadersGraph/Core/Files/MPSGraphUtilities.mm:233:0: error: 'mps.multiply' op operands don't have broadcast-compatible shapes
(mpsFileLoc): /AppleInternal/Library/BuildRoots/4~B5vFugC00QBfsSa38UwjlpDhP2WeLxB3cTjDRaE/Library/Caches/com.apple.xbs/Sources/MetalPerformanceShadersGraph/mpsgraph/MetalPerformanceShadersGraph/Core/Files/MPSGraphUtilities.mm:233:0: note: see current operation: %10 = "mps.multiply"(%arg2, %9) : (tensor<32x50x1xf32>, tensor<32x50xf32>) -> tensor<*xf32>
/AppleInternal/Library/BuildRoots/4~B5vFugC00QBfsSa38UwjlpDhP2WeLxB3cTjDRaE/Library/Caches/com.apple.xbs/Sources/MetalPerformanceShadersGraph/mpsgraph/MetalPerformanceShadersGraph/Core/Files/MPSGraphExecutable.mm:1232: failed assertion `original module failed verification'
Abort