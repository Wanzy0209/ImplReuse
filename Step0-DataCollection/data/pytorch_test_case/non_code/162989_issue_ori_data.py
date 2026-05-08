TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("_fused_sdp_choice", TORCH_FN(_fused_sdp_choice))
}