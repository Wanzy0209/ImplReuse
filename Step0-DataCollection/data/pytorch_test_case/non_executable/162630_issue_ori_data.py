#include <atomic>

// TensorImpl is a target of an intrusive ptr that contains a reference counter.
// in the context of PyTorch, based on my understanding,
// it would be c10::TensorImpl or something c10::TensorImpl holds like ExtraMeta
class TensorImpl : public intrusive_ptr_target<TensorImpl> {
 public:
  ~TensorImpl() {
    // deleting the cached dl managed tensor versioned
    // We need to acquire the value in case it is released by another thread
    // However, because this destructor is triggered as part of the intrusive pointer deletion
    // there is already a memory fence in intrusive pointer deleter triggering to ensure
    // all fields of the TensorImpl are visible here, so we do not have to do acquire, actually 
    // we can even do a non-atomic load here
    DLManagedTensorVersioned* cached = cached_dl_managed_tensor_.load(
      std::memory_order_relaxed);
    if (cached != nullptr) {
      delete cached;
    }
  }
  /*!
   * \brief Converts the current Tensor to a DLPack Tensor.
   * \return The converted DLManagedTensorVersioned pointer.
   */
  DLManagedTensorVersioned* ToDLPack() const {
    // this function holds a strong reference to the TensorImpl
    TensorImpl* self = const_cast<TensorImpl*>(this);
    // we need to use acquire to ensure that write to DLManagedTensorVersioned
    // from another thread is visible to this thread.
    DLManagedTensorVersioned* cached = self->cached_dl_managed_tensor_.load(
      std::memory_order_acquire);
    if (cached == nullptr) {
      // First time conversion: create and populate the DLManagedTensorVersioned.
      // this creation may race among multiple threads, so we need to use atomic exchange.
      DLManagedTensorVersioned* ret = new DLManagedTensorVersioned();
      // Populate metadata (framework-specific logic).
      // Assuming metadata_ is a DLTensor structure
      PopulateMetadata(this, &(self->cached_dl_managed_tensor_->dl_tensor)); 
      // Set the deleter to our custom function.
      ret->deleter = DLManagedTensorDeleter;

      // now set the cached_dl_managed_tensor_ field using CAS
      // success set must release the new value to all other threads
      // failure set must acquire, since the expected value is now coming
      // from another thread that released this value
      if (std::atomic_compare_exchange_strong_explicit(
        &cached_dl_managed_tensor_versioned_, &expected, ret,
        std::memory_order_release, std::memory_order_acquire)) {
        // set is succes
        cached = ret;
      } else {
        // delete the ret value as another thread raced to set this one first
        // expected now contains the value that was set by another thread
        delete ret;
        cached = expected;
      }
    }
    // at this point, cached is the value that officially set to the field
    // Always increment the reference counter of the TensorImpl.
    // This ensures the TensorImpl remains valid as long as the
    // DLPack tensor is in use.
    self->IncRef();
    return cached;
  }

 private:
  // Intrusive reference counter methods.
  void IncRef();
  void DecRef();
  
  // Custom deleter function for DLPack.
  static void DLManagedTensorDeleter(DLManagedTensorVersioned* tensor) {
    // Cast the manager_ctx back to TensorImpl and decrement its reference counter.
    // The TensorImpl and its embedded cache will be deallocated once all
    // references (internal and DLPack) are released.
    static_cast<TensorImpl*>(tensor->manager_ctx)->DecRef();
  }
  
  // Normal tensor metadata, e.g., shape, strides, data pointer.

  // The cached DLManagedTensorVersioned object.
  mutable std::atomic<DLManagedTensorVersioned*> cached_dl_managed_tensor_;
};