diff --git a/test/inductor/test_triton_kernels.py b/test/inductor/test_triton_kernels.py
index eee4dba7f27..2eef80e5976 100644
--- a/test/inductor/test_triton_kernels.py
+++ b/test/inductor/test_triton_kernels.py
@@ -1216,6 +1216,20 @@ def forward(self, x_1, output_1):
         compiled_out = torch.compile(f)(x, y)
         self.assertEqual(compiled_out, eager_out)
 
+    @requires_gpu
+    def test_triton_kernel_to_cpu(self):
+        def f(x, y):
+            out = torch.zeros_like(x)
+            add_kernel[(1,)](x, y, out, 16, 16)
+            out_cpu = out.cpu() + 1
+            return out_cpu
+
+        x = torch.randn(4, 4, device=GPU_TYPE)
+        y = torch.randn(4, 4, device=GPU_TYPE)
+        eager_out = f(x, y)
+        compiled_out = torch.compile(f)(x, y)
+        self.assertEqual(compiled_out, eager_out)
+
     @requires_gpu
     def test_triton_kernel_out_of_order(self):
         @triton.jit