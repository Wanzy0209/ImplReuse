class CustomProcessGroupNCCL: public ProcessGroupNCCL {
public:
using ProcessGroupNCCL::ProcessGroupNCCL;
  
// Set NCCL stream by device index
void setNcclStreamByDevice(int device_index, const c10::cuda::CUDAStream& stream) {
  std::string key = std::to_string(device_index);
  std::lock_guard<std::mutex> guard(mutex_);
  if (ncclStreams_.find(key) != ncclStreams_.end()) {
    ncclStreams_.at(key) = stream;
  } else {
    TORCH_CHECK(false, "Key not found in ncclStreams_: ", key);
  }
}
  
// Get NCCL stream by device index
c10::cuda::CUDAStream getNcclStreamByDevice(int device_index) {
  std::string key = std::to_string(device_index);
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = ncclStreams_.find(key);
  TORCH_CHECK(it != ncclStreams_.end(), "Key not found in ncclStreams_: ", key);
  return it->second;
}
  
// List all available NCCL stream keys
std::vector<std::string> getAvailableStreamKeys() {
  std::lock_guard<std::mutex> guard(mutex_);
  std::vector<std::string> keys;
  for (const auto& pair : ncclStreams_) {
    keys.push_back(pair.first);
  }
  return keys;
}

// List all available NCCL streams 
std::unordered_map<std::string, c10::cuda::CUDAStream> getAvailableStreams(){
  return ncclStreams_;
} 
  
protected:
std::mutex mutex_;
};