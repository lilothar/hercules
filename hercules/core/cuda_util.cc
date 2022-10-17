
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "hercules/core/cuda_util.h"
#include "hercules/common/error_code.h"
#include "hercules/common/nvtx.h"

namespace hercules::core {

#ifdef HERCULES_ENABLE_GPU
    void CUDART_CB
MemcpyHost(void* args)
{
  auto* copy_params = reinterpret_cast<CopyParams*>(args);
  memcpy(copy_params->dst_, copy_params->src_, copy_params->byte_size_);
  delete copy_params;
}
#endif  // HERCULES_ENABLE_GPU

    flare::result_status
    GetDeviceMemoryInfo(const int device_id, size_t *free, size_t *total) {
        *free = 0;
        *total = 0;
#ifdef HERCULES_ENABLE_GPU
        // Make sure that correct device is set before creating stream and
  // then restore the device to what was set by the caller.
  int current_device;
  auto cuerr = cudaGetDevice(&current_device);
  bool overridden = false;
  if (cuerr == cudaSuccess) {
    overridden = (current_device != device_id);
    if (overridden) {
      cuerr = cudaSetDevice(device_id);
    }
  }

  if (cuerr == cudaSuccess) {
    cuerr = cudaMemGetInfo(free, total);
  }

  if (overridden) {
    cudaSetDevice(current_device);
  }

  if (cuerr != cudaSuccess) {
    return flare::result_status(
        hercules::common::ERROR_INTERNAL,
        (std::string("unable to get memory info for device ") +
         std::to_string(device_id) + ": " + cudaGetErrorString(cuerr)));
  }
#endif  // HERCULES_ENABLE_GPU
        return flare::result_status::success();
    }

    flare::result_status
    EnablePeerAccess(const double min_compute_capability) {
#ifdef HERCULES_ENABLE_GPU
        // If we can't enable peer access for one device pair, the best we can
  // do is skipping it...
  std::set<int> supported_gpus;
  bool all_enabled = false;
  if (GetSupportedGPUs(&supported_gpus, min_compute_capability).IsOk()) {
    all_enabled = true;
    int can_access_peer = false;
    for (const auto& host : supported_gpus) {
      auto cuerr = cudaSetDevice(host);

      if (cuerr == cudaSuccess) {
        for (const auto& peer : supported_gpus) {
          if (host == peer) {
            continue;
          }

          cuerr = cudaDeviceCanAccessPeer(&can_access_peer, host, peer);
          if ((cuerr == cudaSuccess) && (can_access_peer == 1)) {
            cuerr = cudaDeviceEnablePeerAccess(peer, 0);
          }

          all_enabled &= ((cuerr == cudaSuccess) && (can_access_peer == 1));
        }
      }
    }
  }
  if (!all_enabled) {
    return flare::result_status(
        hercules::common::ERROR_UNSUPPORTED,
        "failed to enable peer access for some device pairs");
  }
#endif  // HERCULES_ENABLE_GPU
        return flare::result_status::success();
    }

    flare::result_status
    CopyBuffer(
            const std::string &msg, const hercules::proto::MemoryType src_memory_type,
            const int64_t src_memory_type_id,
            const hercules::proto::MemoryType dst_memory_type,
            const int64_t dst_memory_type_id, const size_t byte_size, const void *src,
            void *dst, cudaStream_t cuda_stream, bool *cuda_used, bool copy_on_stream) {
        NVTX_RANGE(nvtx_, "CopyBuffer");

        *cuda_used = false;

        // For CUDA memcpy, all host to host copy will be blocked in respect to the
        // host, so use memcpy() directly. In this case, need to be careful on whether
        // the src buffer is valid.
        if ((src_memory_type != hercules::proto::MEMORY_GPU) &&
            (dst_memory_type != hercules::proto::MEMORY_GPU)) {
#ifdef HERCULES_ENABLE_GPU
            if (copy_on_stream) {
      auto params = new CopyParams(dst, src, byte_size);
      cudaLaunchHostFunc(
          cuda_stream, MemcpyHost, reinterpret_cast<void*>(params));
      *cuda_used = true;
    } else {
      memcpy(dst, src, byte_size);
    }
#else
            memcpy(dst, src, byte_size);
#endif  // HERCULES_ENABLE_GPU
        } else {
#ifdef HERCULES_ENABLE_GPU
            RETURN_IF_CUDA_ERR(
        cudaMemcpyAsync(dst, src, byte_size, cudaMemcpyDefault, cuda_stream),
        msg + ": failed to perform CUDA copy");

    *cuda_used = true;
#else
            return flare::result_status(
                    hercules::common::ERROR_INTERNAL,
                    msg + ": try to use CUDA copy while GPU is not supported");
#endif  // HERCULES_ENABLE_GPU
        }

        return flare::result_status::success();
    }

    void
    CopyBufferHandler(
            const std::string &msg, const hercules::proto::MemoryType src_memory_type,
            const int64_t src_memory_type_id,
            const hercules::proto::MemoryType dst_memory_type,
            const int64_t dst_memory_type_id, const size_t byte_size, const void *src,
            void *dst, cudaStream_t cuda_stream, void *response_ptr,
            hercules::common::sync_queue <std::tuple<flare::result_status, bool, void *>> *
            completion_queue) {
        bool cuda_used = false;
        flare::result_status status = CopyBuffer(
                msg, src_memory_type, src_memory_type_id, dst_memory_type,
                dst_memory_type_id, byte_size, src, dst, cuda_stream, &cuda_used);
        completion_queue->Put(std::make_tuple(status, cuda_used, response_ptr));
    }

#ifdef HERCULES_ENABLE_GPU
    flare::result_status
CheckGPUCompatibility(const int gpu_id, const double min_compute_capability)
{
  // Query the compute capability from the device
  cudaDeviceProp cuprops;
  cudaError_t cuerr = cudaGetDeviceProperties(&cuprops, gpu_id);
  if (cuerr != cudaSuccess) {
    return flare::result_status(
        hercules::common::ERROR_INTERNAL,
        "unable to get CUDA device properties for GPU ID" +
            std::to_string(gpu_id) + ": " + cudaGetErrorString(cuerr));
  }

  double compute_compability = cuprops.major + (cuprops.minor / 10.0);
  if ((compute_compability > min_compute_capability) ||
      (abs(compute_compability - min_compute_capability) < 0.01)) {
    return flare::result_status::success();
  } else {
    return flare::result_status(
        hercules::common::ERROR_UNSUPPORTED,
        "gpu " + std::to_string(gpu_id) + " has compute capability '" +
            std::to_string(cuprops.major) + "." +
            std::to_string(cuprops.minor) +
            "' which is less than the minimum supported of '" +
            std::to_string(min_compute_capability) + "'");
  }
}

flare::result_status
GetSupportedGPUs(
    std::set<int>* supported_gpus, const double min_compute_capability)
{
  // Make sure set is empty before starting
  supported_gpus->clear();

  int device_cnt;
  cudaError_t cuerr = cudaGetDeviceCount(&device_cnt);
  if ((cuerr == cudaErrorNoDevice) || (cuerr == cudaErrorInsufficientDriver)) {
    device_cnt = 0;
  } else if (cuerr != cudaSuccess) {
    return flare::result_status(
        hercules::common::ERROR_INTERNAL, "unable to get number of CUDA devices: " +
                                    std::string(cudaGetErrorString(cuerr)));
  }

  // populates supported_gpus
  for (int gpu_id = 0; gpu_id < device_cnt; gpu_id++) {
    flare::result_status status = CheckGPUCompatibility(gpu_id, min_compute_capability);
    if (status.IsOk()) {
      supported_gpus->insert(gpu_id);
    }
  }
  return flare::result_status::success();
}

flare::result_status
SupportsIntegratedZeroCopy(const int gpu_id, bool* zero_copy_support)
{
  // Query the device to check if integrated
  cudaDeviceProp cuprops;
  cudaError_t cuerr = cudaGetDeviceProperties(&cuprops, gpu_id);
  if (cuerr != cudaSuccess) {
    return flare::result_status(
        hercules::common::ERROR_INTERNAL,
        "unable to get CUDA device properties for GPU ID" +
            std::to_string(gpu_id) + ": " + cudaGetErrorString(cuerr));
  }

  // Zero-copy supported only on integrated GPU when it can map host memory
  if (cuprops.integrated && cuprops.canMapHostMemory) {
    *zero_copy_support = true;
  } else {
    *zero_copy_support = false;
  }

  return flare::result_status::success();
}

#endif

}
