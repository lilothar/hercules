
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "hercules/core/buffer_attributes.h"
#include "hercules/core/constants.h"

namespace hercules::core {
    void
    buffer_attributes::set_byte_size(const size_t &byte_size) {
        byte_size_ = byte_size;
    }

    void
    buffer_attributes::set_memory_type(const hercules::proto::MemoryType &memory_type) {
        memory_type_ = memory_type;
    }

    void
    buffer_attributes::set_memory_type_id(const int64_t &memory_type_id) {
        memory_type_id_ = memory_type_id;
    }

    void
    buffer_attributes::set_cuda_ipc_handle(void *cuda_ipc_handle) {
        char *lcuda_ipc_handle = reinterpret_cast<char *>(cuda_ipc_handle);
        cuda_ipc_handle_.clear();
        std::copy(
                lcuda_ipc_handle, lcuda_ipc_handle + kCudaIpcStructSize,
                std::back_inserter(cuda_ipc_handle_));
    }

    void *
    buffer_attributes::cuda_ipc_handle() {
        if (cuda_ipc_handle_.empty()) {
            return nullptr;
        } else {
            return reinterpret_cast<void *>(cuda_ipc_handle_.data());
        }
    }

    size_t
    buffer_attributes::byte_size() const {
        return byte_size_;
    }

    hercules::proto::MemoryType
    buffer_attributes::memory_type() const {
        return memory_type_;
    }

    int64_t
    buffer_attributes::memory_type_id() const {
        return memory_type_id_;
    }

    buffer_attributes::buffer_attributes(
            size_t byte_size, hercules::proto::MemoryType memory_type,
            int64_t memory_type_id, char *cuda_ipc_handle)
            : byte_size_(byte_size), memory_type_(memory_type),
              memory_type_id_(memory_type_id) {
        // cuda ipc handle size
        cuda_ipc_handle_.reserve(kCudaIpcStructSize);

        if (cuda_ipc_handle != nullptr) {
            std::copy(
                    cuda_ipc_handle, cuda_ipc_handle + kCudaIpcStructSize,
                    std::back_inserter(cuda_ipc_handle_));
        }
    }
}  // namespace hercules::core