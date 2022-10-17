//
// Created by liyinbin on 2022/10/13.
//

#include "hercules/core/pinned_memory_manager.h"

#include <sstream>
#include "hercules/core/numa_util.h"
#include "hercules/common/error_code.h"
#include <flare/log/logging.h>

#ifdef HERCULES_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // HERCULES_ENABLE_GPU

namespace hercules::core {

    namespace {

        std::string
        PointerToString(void *ptr) {
            std::stringstream ss;
            ss << ptr;
            return ss.str();
        }

        flare::result_status
        ParseIntOption(const std::string &msg, const std::string &arg, int *value) {
            try {
                *value = std::stoi(arg);
            }
            catch (const std::invalid_argument &ia) {
                return flare::result_status(
                        hercules::common::ERROR_INVALID_ARG,
                        msg + ": Can't parse '" + arg + "' to integer");
            }
            return flare::result_status::success();
        }

    }  // namespace

    std::unique_ptr<pinned_memory_manager> pinned_memory_manager::instance_;
    uint64_t pinned_memory_manager::pinned_memory_byte_size_;

    pinned_memory_manager::pinned_memory::pinned_memory(
            void *pinned_memory_buffer, uint64_t size)
            : pinned_memory_buffer_(pinned_memory_buffer) {
        if (pinned_memory_buffer_ != nullptr) {
            managed_pinned_memory_ = boost::interprocess::managed_external_buffer(
                    boost::interprocess::create_only_t{}, pinned_memory_buffer_, size);

        }
    }


    pinned_memory_manager::pinned_memory::~pinned_memory() {
#ifdef HERCULES_ENABLE_GPU
        if (pinned_memory_buffer_ != nullptr) {
            cudaFreeHost(pinned_memory_buffer_);
        }
#endif  // HERCULES_ENABLE_GPU
    }

    pinned_memory_manager::~pinned_memory_manager() {
        // Clean up
        for (const auto &memory_info : memory_info_) {
            const auto &is_pinned = memory_info.second.first;
            if (!is_pinned) {
                free(memory_info.first);
            }
        }
    }

    void
    pinned_memory_manager::add_pinned_memory_buffer(
            const std::shared_ptr<pinned_memory> &pinned_memory_buffer,
            unsigned long node_mask) {
        pinned_memory_buffers_[node_mask] = pinned_memory_buffer;
    }

    flare::result_status
    pinned_memory_manager::alloc_internal(
            void **ptr, uint64_t size, hercules::proto::MemoryType *allocated_type,
            bool allow_nonpinned_fallback, pinned_memory *pinned_memory_buffer) {
        auto status = flare::result_status::success();
        if (pinned_memory_buffer->pinned_memory_buffer_ != nullptr) {
            std::lock_guard<std::mutex> lk(pinned_memory_buffer->buffer_mtx_);
            *ptr = pinned_memory_buffer->managed_pinned_memory_.allocate(size, std::nothrow_t{});
            *allocated_type = hercules::proto::MEMORY_CPU_BINDING;
            if (*ptr == nullptr) {
                status = flare::result_status(
                        hercules::common::ERROR_INTERNAL, "failed to allocate pinned system memory");
            }
        } else {
            status = flare::result_status(
                    hercules::common::ERROR_INTERNAL,
                    "failed to allocate pinned system memory: no pinned memory pool");
        }

        bool is_pinned = true;
        if ((!status.is_ok()) && allow_nonpinned_fallback) {
            static bool warning_logged = false;
            if (!warning_logged) {
                FLARE_LOG(WARNING) << status
                                   << ", falling back to non-pinned system memory";
                warning_logged = true;
            }
            *ptr = malloc(size);
            *allocated_type = hercules::proto::MEMORY_CPU;
            is_pinned = false;
            if (*ptr == nullptr) {
                status = flare::result_status(
                        hercules::common::ERROR_INTERNAL,
                        "failed to allocate non-pinned system memory");
            } else {
                status = flare::result_status::success();
            }
        }

        // keep track of allocated buffer or clean up
        {
            std::lock_guard<std::mutex> lk(info_mtx_);
            if (status.is_ok()) {
                auto res = memory_info_.emplace(
                        *ptr, std::make_pair(is_pinned, pinned_memory_buffer));
                if (!res.second) {
                    status = flare::result_status(
                            hercules::common::ERROR_INTERNAL, "unexpected memory address collision, '" +
                                                              PointerToString(*ptr) +
                                                              "' has been managed");
                }
                FLARE_LOG(INFO) << (is_pinned ? "" : "non-")
                                << "pinned memory allocation: "
                                << "size " << size << ", addr " << *ptr;
            }
        }

        if ((!status.is_ok()) && (*ptr != nullptr)) {
            if (is_pinned) {
                std::lock_guard<std::mutex> lk(pinned_memory_buffer->buffer_mtx_);
                pinned_memory_buffer->managed_pinned_memory_.deallocate(*ptr);
            } else {
                free(*ptr);
            }
        }

        return status;
    }

    flare::result_status
    pinned_memory_manager::free_internal(void *ptr) {
        bool is_pinned = true;
        pinned_memory *pinned_memory_buffer = nullptr;
        {
            std::lock_guard<std::mutex> lk(info_mtx_);
            auto it = memory_info_.find(ptr);
            if (it != memory_info_.end()) {
                is_pinned = it->second.first;
                pinned_memory_buffer = it->second.second;
                FLARE_LOG(INFO) << (is_pinned ? "" : "non-")
                                << "pinned memory deallocation: "
                                << "addr " << ptr;
                memory_info_.erase(it);
            } else {
                return flare::result_status(
                        hercules::common::ERROR_INTERNAL, "unexpected memory address '" +
                                                          PointerToString(ptr) +
                                                          "' is not being managed");
            }
        }

        if (is_pinned) {
            std::lock_guard<std::mutex> lk(pinned_memory_buffer->buffer_mtx_);
            pinned_memory_buffer->managed_pinned_memory_.deallocate(ptr);
        } else {
            free(ptr);
        }
        return flare::result_status::success();
    }

    void
    pinned_memory_manager::reset() {
        instance_.reset();
    }

    flare::result_status
    pinned_memory_manager::create(const options &options) {
        if (instance_ != nullptr) {
            FLARE_LOG(WARNING) << "New pinned memory pool of size "
                               << options.pinned_memory_pool_byte_size_
                               << " could not be created since one already exists"
                               << " of size " << pinned_memory_byte_size_;
            return flare::result_status::success();
        }

        instance_.reset(new pinned_memory_manager());
        if (options.host_policy_map_.empty()) {
            void *buffer = nullptr;
#ifdef HERCULES_ENABLE_GPU
            auto err = cudaHostAlloc(
    &buffer, options.pinned_memory_pool_byte_size_, cudaHostAllocPortable);
if (err != cudaSuccess) {
  buffer = nullptr;
  LOG_WARNING << "Unable to allocate pinned system memory, pinned memory "
                 "pool will not be available: "
              << std::string(cudaGetErrorString(err));
} else if (options.pinned_memory_pool_byte_size_ != 0) {
  LOG_INFO << "Pinned memory pool is created at '"
           << PointerToString(buffer) << "' with size "
           << options.pinned_memory_pool_byte_size_;
} else {
  LOG_INFO << "Pinned memory pool disabled";
}
#endif  // HERCULES_ENABLE_GPU
            try {
                instance_->add_pinned_memory_buffer(
                        std::shared_ptr<pinned_memory>(
                                new pinned_memory(buffer, options.pinned_memory_pool_byte_size_)),
                        0);
            }
            catch (const std::exception &ex) {
                return flare::result_status(
                        hercules::common::ERROR_INTERNAL,
                        "Failed to add Pinned Memory buffer: " + std::string(ex.what()));
            }
        } else {
            // Create only one buffer / manager should be created for one node,
            // and all associated devices should request memory from the shared manager
            std::map<int32_t, std::string> numa_map;
            for (const auto &host_policy : options.host_policy_map_) {
                const auto numa_it = host_policy.second.find("numa-node");
                if (numa_it != host_policy.second.end()) {
                    int32_t numa_id;
                    if (ParseIntOption("Parsing NUMA node", numa_it->second, &numa_id)
                            .is_ok()) {
                        numa_map.emplace(numa_id, host_policy.first);
                    }
                }
            }
            for (const auto &node_policy : numa_map) {
                auto status =
                        set_numa_memory_policy(options.host_policy_map_.at(node_policy.second));
                if (!status.is_ok()) {
                    FLARE_LOG(WARNING) << "Unable to allocate pinned system memory for NUMA node "
                                       << node_policy.first << ": " << status;
                    continue;
                }
                unsigned long node_mask;
                status = get_numa_memory_policy_node_mask(&node_mask);
                if (!status.is_ok()) {
                    FLARE_LOG(WARNING) << "Unable to get NUMA node set for current thread: "
                                       << status;
                    continue;
                }
                void *buffer = nullptr;
#ifdef HERCULES_ENABLE_GPU
                auto err = cudaHostAlloc(
      &buffer, options.pinned_memory_pool_byte_size_,
      cudaHostAllocPortable);
  if (err != cudaSuccess) {
    buffer = nullptr;
    LOG_WARNING << "Unable to allocate pinned system memory, pinned memory "
                   "pool will not be available: "
                << std::string(cudaGetErrorString(err));
  } else if (options.pinned_memory_pool_byte_size_ != 0) {
    LOG_INFO << "Pinned memory pool is created at '"
             << PointerToString(buffer) << "' with size "
             << options.pinned_memory_pool_byte_size_;
  } else {
    LOG_INFO << "Pinned memory pool disabled";
  }
#endif  // HERCULES_ENABLE_GPU
                reset_numa_memory_policy();
                try {
                    instance_->add_pinned_memory_buffer(
                            std::shared_ptr<pinned_memory>(new pinned_memory(
                                    buffer, options.pinned_memory_pool_byte_size_)),
                            node_mask);
                }
                catch (const std::exception &ex) {
                    return flare::result_status(
                            hercules::common::ERROR_INTERNAL,
                            "Failed to add Pinned Memory buffer with host policy: " +
                            std::string(ex.what()));
                }
            }
            // If no pinned memory is allocated, add an empty entry where all allocation
            // will be on normal system memory
            if (instance_->pinned_memory_buffers_.empty()) {
                try {
                    instance_->add_pinned_memory_buffer(
                            std::shared_ptr<pinned_memory>(new pinned_memory(
                                    nullptr, options.pinned_memory_pool_byte_size_)),
                            0);
                }
                catch (const std::exception &ex) {
                    return flare::result_status(
                            hercules::common::ERROR_INTERNAL,
                            "Failed to add empty Pinned Memory entry: " +
                            std::string(ex.what()));
                }
            }
        }
        pinned_memory_byte_size_ = options.pinned_memory_pool_byte_size_;
        return flare::result_status::success();
    }

    flare::result_status
    pinned_memory_manager::alloc(
            void **ptr, uint64_t size, hercules::proto::MemoryType *allocated_type,
            bool allow_nonpinned_fallback) {
        if (instance_ == nullptr) {
            return flare::result_status(
                    hercules::common::ERROR_UNAVAILABLE, "pinned_memory_manager has not been created");
        }

        auto pinned_memory_buffer =
                instance_->pinned_memory_buffers_.begin()->second.get();
        if (instance_->pinned_memory_buffers_.size() > 1) {
            unsigned long node_mask;
            if (get_numa_memory_policy_node_mask(&node_mask).is_ok()) {
                auto it = instance_->pinned_memory_buffers_.find(node_mask);
                if (it != instance_->pinned_memory_buffers_.end()) {
                    pinned_memory_buffer = it->second.get();
                }
            }
        }

        return instance_->alloc_internal(
                ptr, size, allocated_type, allow_nonpinned_fallback,
                pinned_memory_buffer);
    }

    flare::result_status
    pinned_memory_manager::Free(void *ptr) {
        if (instance_ == nullptr) {
            return flare::result_status(
                    hercules::common::ERROR_UNAVAILABLE, "pinned_memory_manager has not been created");
        }

        return instance_->free_internal(ptr);
    }

}  // namespace hercules::core