namespace torch::stable {
namespace {
template <typename T>
StableIValue from(T val);
template <typename T>
T to(StableIValue val);
}
}