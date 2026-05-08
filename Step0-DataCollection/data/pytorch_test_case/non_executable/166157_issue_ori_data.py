namespace c10::openreg {

class OpenRegDeviceAllocator final : public c10::DeviceAllocator {
 private:
  c10::CachingDeviceAllocator::DeviceStats stats_;
  bool initialized_ = false;

 public:
  OpenRegDeviceAllocator() = default;

  at::DataPtr allocate(size_t nbytes) override {
    //...
    
    void* data = nullptr;
    if (nbytes > 0) {
      orMalloc(&data, nbytes);
      TORCH_CHECK(data, "Failed to allocate ", nbytes, " bytes on openreg device.");
      
      stats_.allocated_bytes[0].increase(nbytes);
      stats_.reserved_bytes[0].increase(nbytes);
    }
    
    auto curr_device = c10::Device(c10::DeviceType::PrivateUse1, current_device_index);
    return {data, data, &ReportAndDelete, curr_device};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    orMemcpy(dest, src, count, orMemcpyDeviceToDevice);
  }

  bool initialized() override {
    return initialized_;
  }

  void emptyCache(MempoolId_t mempool_id = {0, 0}) override {
    // TODO: for further development of memory caching
  }

  void recordStream(const DataPtr& ptr, c10::Stream stream) override {
    // TODO: for further development of memory caching
  }

  c10::CachingDeviceAllocator::DeviceStats getDeviceStats(
      c10::DeviceIndex device) override {
    return stats_;
  }

  void resetAccumulatedStats(c10::DeviceIndex device) override {
    for (size_t i = 0; i < stats_.allocated_bytes.size(); ++i) {
      stats_.allocated_bytes[i].reset_accumulated();
      stats_.reserved_bytes[i].reset_accumulated();
    }
    stats_.num_alloc_retries = 0;
  }

  void resetPeakStats(c10::DeviceIndex device) override {
    for (size_t i = 0; i < stats_.allocated_bytes.size(); ++i) {
      stats_.allocated_bytes[i].reset_peak();
      stats_.reserved_bytes[i].reset_peak();
    }
  }

 private:
  static void ReportAndDelete(void* ptr) {
    if (!ptr) return;
    orFreeHost(ptr);
  }
};

// ...

}