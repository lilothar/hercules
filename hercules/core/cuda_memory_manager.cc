
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/
#ifdef HERCULES_ENABLE_GPU

#include <sstream>
#include <set>
#include "hercules/core/cuda_memory_manager.h"
#include "hercules/core/cuda_util.h"
#include "hercules/common/cnmem.h"
#include "hercules/common/error_code.h"
#include <flare/log/logging.h>

namespace {

#define RETURN_IF_CNMEM_ERROR(S, MSG)                    \
  do {                                                   \
    auto status__ = (S);                                 \
    if (status__ != CNMEM_STATUS_SUCCESS) {              \
      return flare::result_status(                                     \
          hercules::common::ERROR_INTERNAL,                        \
          (MSG) + ": " + cnmemGetErrorString(status__)); \
    }                                                    \
  } while (false)

    std::string
    PointerToString(void* ptr)
    {
        std::stringstream ss;
        ss << ptr;
        return ss.str();
    }

}  // namespace

namespace hercules::core {


    std::unique_ptr<cuda_memory_manager> cuda_memory_manager::instance_;
    std::mutex cuda_memory_manager::instance_mu_;

    cuda_memory_manager::~cuda_memory_manager()
    {
        if (has_allocation_) {
            auto status = cnmemFinalize();
            if (status != CNMEM_STATUS_SUCCESS) {
                FLARE_LOG(ERROR) << "Failed to finalize CUDA memory manager: [" << status << "] "
                          << cnmemGetErrorString(status);
            }
        }
    }

    void
    cuda_memory_manager::reset()
    {
        std::lock_guard<std::mutex> lock(instance_mu_);
        instance_.reset();
    }

    flare::result_status
    cuda_memory_manager::Create(const cuda_memory_manager::options& options)
    {
        // Ensure thread-safe creation of CUDA memory pool
        std::lock_guard<std::mutex> lock(instance_mu_);
        if (instance_ != nullptr) {
            FLARE_LOG(WARNING) << "New CUDA memory pools could not be created since they "
                           "already exists";
            return flare::result_status::success();
        }

        std::set<int> supported_gpus;
        auto status = GetSupportedGPUs(
                &supported_gpus, options.min_supported_compute_capability_);
        if (status.IsOk()) {
            std::vector<cnmemDevice_t> devices;
            for (auto gpu : supported_gpus) {
                const auto it = options.memory_pool_byte_size_.find(gpu);
                if ((it != options.memory_pool_byte_size_.end()) && (it->second != 0)) {
                    devices.emplace_back();
                    auto& device = devices.back();
                    memset(&device, 0, sizeof(device));
                    device.device = gpu;
                    device.size = it->second;

                    FLARE_LOG(INFO) << "CUDA memory pool is created on device " << device.device
                             << " with size " << device.size;
                }
            }

            if (!devices.empty()) {
                RETURN_IF_CNMEM_ERROR(
                        cnmemInit(devices.size(), devices.data(), CNMEM_FLAGS_CANNOT_GROW),
                        std::string("Failed to finalize CUDA memory manager"));
            } else {
                FLARE_LOG(INFO) << "CUDA memory pool disabled";
            }

            // Use to finalize CNMeM properly when out of scope
            instance_.reset(new cuda_memory_manager(!devices.empty()));
        } else {
            return Status(
                    status.ErrorCode(),
                    "Failed to initialize CUDA memory manager: " + status.Message());
        }

        return flare::result_status::success();
    }

    flare::result_status
    cuda_memory_manager::alloc(void** ptr, uint64_t size, int64_t device_id)
    {
        if (instance_ == nullptr) {
            return flare::result_status(
                    hercules::common::ERROR_UNAVAILABLE, "cuda_memory_manager has not been created");
        } else if (!instance_->has_allocation_) {
            return flare::result_status(
                    hercules::common::ERROR_UNAVAILABLE,
                    "cuda_memory_manager has no preallocated CUDA memory");
        }

        int current_device;
        RETURN_IF_CUDA_ERR(
                cudaGetDevice(&current_device), std::string("Failed to get device"));
        bool overridden = (current_device != device_id);
        if (overridden) {
            RETURN_IF_CUDA_ERR(
                    cudaSetDevice(device_id), std::string("Failed to set device"));
        }

        // Defer returning error to make sure the device is recovered
        auto err = cnmemMalloc(ptr, size, nullptr);

        if (overridden) {
            cudaSetDevice(current_device);
        }

        RETURN_IF_CNMEM_ERROR(
                err, std::string("Failed to allocate CUDA memory with byte size ") +
                     std::to_string(size) + " on GPU " + std::to_string(device_id));
        return flare::result_status::success();
    }

    flare::result_status
    cuda_memory_manager::free(void* ptr, int64_t device_id)
    {
        if (instance_ == nullptr) {
            return flare::result_status(
                    hercules::common::ERROR_UNAVAILABLE, "cuda_memory_manager has not been created");
        } else if (!instance_->has_allocation_) {
            return flare::result_status(
                    hercules::common::ERROR_UNAVAILABLE,
                    "cuda_memory_manager has no preallocated CUDA memory");
        }

        int current_device;
        RETURN_IF_CUDA_ERR(
                cudaGetDevice(&current_device), std::string("Failed to get device"));
        bool overridden = (current_device != device_id);
        if (overridden) {
            RETURN_IF_CUDA_ERR(
                    cudaSetDevice(device_id), std::string("Failed to set device"));
        }

        // Defer returning error to make sure the device is recovered
        auto err = cnmemFree(ptr, nullptr);

        if (overridden) {
            cudaSetDevice(current_device);
        }

        RETURN_IF_CNMEM_ERROR(
                err, std::string("Failed to deallocate CUDA memory at address ") +
                     PointerToString(ptr) + " on GPU " + std::to_string(device_id));
        return flare::result_status::success();
    }

}  // namespace hercules::core

#endif  // HERCULES_ENABLE_GPU