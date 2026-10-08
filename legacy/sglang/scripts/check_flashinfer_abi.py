#!/usr/bin/env python3
"""Host-only check of the exact owning-Tensor ABI adapter; no CUDA speed claim."""
from pathlib import Path
import subprocess
import tvm_ffi

root=Path(__file__).resolve().parents[1]
include=Path(tvm_ffi.__file__).parent/'include'
code='''#include "legacy_tensor_compat.h"
#include <tvm/ffi/function.h>
static_assert(std::is_default_constructible_v<flashinfer_ffi::Tensor>);
void check(flashinfer_ffi::Tensor t) {
  auto n=t->ndim; auto data=t->data; auto dev=t->device.device_id;
  auto dtype=t->dtype.bits; auto shape=t->shape[0]; auto stride=t->strides[0];
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(check,check);
'''
subprocess.run(['c++','-std=c++17','-fsyntax-only','-x','c++','-I'+str(include),
                '-I'+str(root/'patches'),'-'],input=code,text=True,check=True)
print('Host Tensor accessor and typed function registration: PASS (not CUDA JIT validation)')
