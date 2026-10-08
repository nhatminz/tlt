// FlashInfer0.4.0 expects a typed Tensor::operator->; stable TVM-FFI0.1
// returns Object* there. Restore the legacy accessor without tensor copies,
// changes to allocation ownership, or changes to CUDA kernel math.
#pragma once
#include <tvm/ffi/container/tensor.h>
namespace flashinfer_ffi {
using namespace tvm::ffi;
class Tensor : public tvm::ffi::Tensor {
 public:
  using tvm::ffi::Tensor::Tensor;
  Tensor() = default;
  Tensor(tvm::ffi::Tensor value) : tvm::ffi::Tensor(std::move(value)) {}
  const ContainerType* operator->() const { return get(); }
};
}  // namespace flashinfer_ffi
