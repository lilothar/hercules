//
// Created by liyinbin on 2022/10/12.
//

#include "hercules/core/memory_base.h"
#include "hercules/core/pinned_memory_manager.h"
#include "hercules/core/cuda_memory_manager.h"
#include <flare/log/logging.h>

namespace hercules::core {

    memory_reference::memory_reference() : memory_base() {}

    const char *
    memory_reference::buffer_at(
            size_t idx, size_t *byte_size, hercules::proto::MemoryType *memory_type,
            int64_t *memory_type_id) const {
        if (idx >= buffer_.size()) {
            *byte_size = 0;
            *memory_type = hercules::proto::MEMORY_CPU;
            *memory_type_id = 0;
            return nullptr;
        }
        *memory_type = buffer_[idx].buffer_attributes_.memory_type();
        *memory_type_id = buffer_[idx].buffer_attributes_.memory_type_id();
        *byte_size = buffer_[idx].buffer_attributes_.byte_size();
        return buffer_[idx].buffer_;
    }

    const char *
    memory_reference::buffer_at(size_t idx, buffer_attributes **buffer_attributes) {
        if (idx >= buffer_.size()) {
            *buffer_attributes = nullptr;
            return nullptr;
        }

        *buffer_attributes = &(buffer_[idx].buffer_attributes_);
        return buffer_[idx].buffer_;
    }

    size_t
    memory_reference::add_buffer(
            const char *buffer, size_t byte_size, hercules::proto::MemoryType memory_type,
            int64_t memory_type_id) {
        total_byte_size_ += byte_size;
        buffer_count_++;
        buffer_.emplace_back(buffer, byte_size, memory_type, memory_type_id);
        return buffer_.size() - 1;
    }

    size_t
    memory_reference::add_buffer(
            const char *buffer, buffer_attributes *buffer_attributes) {
        total_byte_size_ += buffer_attributes->byte_size();
        buffer_count_++;
        buffer_.emplace_back(buffer, buffer_attributes);
        return buffer_.size() - 1;
    }

    size_t
    memory_reference::add_buffer_front(
            const char *buffer, size_t byte_size, hercules::proto::MemoryType memory_type,
            int64_t memory_type_id) {
        total_byte_size_ += byte_size;
        buffer_count_++;
        buffer_.emplace(
                buffer_.begin(), buffer, byte_size, memory_type, memory_type_id);
        return buffer_.size() - 1;
    }


    mutable_memory::mutable_memory(
            char *buffer, size_t byte_size, hercules::proto::MemoryType memory_type,
            int64_t memory_type_id)
            : memory_base(), buffer_(buffer),
              buffer_attributes_(
                      buffer_attributes(byte_size, memory_type, memory_type_id, nullptr)) {
        total_byte_size_ = byte_size;
        buffer_count_ = (byte_size == 0) ? 0 : 1;
    }

    const char *
    mutable_memory::buffer_at(
            size_t idx, size_t *byte_size, hercules::proto::MemoryType *memory_type,
            int64_t *memory_type_id) const {
        if (idx != 0) {
            *byte_size = 0;
            *memory_type = hercules::proto::MEMORY_CPU;
            *memory_type_id = 0;
            return nullptr;
        }
        *byte_size = total_byte_size_;
        *memory_type = buffer_attributes_.memory_type();
        *memory_type_id = buffer_attributes_.memory_type_id();
        return buffer_;
    }

    const char *
    mutable_memory::buffer_at(size_t idx, buffer_attributes **buffer_attributes) {
        if (idx != 0) {
            *buffer_attributes = nullptr;
            return nullptr;
        }

        *buffer_attributes = &buffer_attributes_;
        return buffer_;
    }

    char *
    mutable_memory::mutable_buffer(
            hercules::proto::MemoryType *memory_type, int64_t *memory_type_id) {
        if (memory_type != nullptr) {
            *memory_type = buffer_attributes_.memory_type();
        }
        if (memory_type_id != nullptr) {
            *memory_type_id = buffer_attributes_.memory_type_id();
        }

        return buffer_;
    }


    allocated_memory::allocated_memory(
            size_t byte_size, hercules::proto::MemoryType memory_type,
            int64_t memory_type_id)
            : mutable_memory(nullptr, byte_size, memory_type, memory_type_id) {
        if (total_byte_size_ != 0) {
            // Allocate memory with the following fallback policy:
            // CUDA memory -> pinned system memory -> non-pinned system memory
            switch (buffer_attributes_.memory_type()) {
#ifdef HERCULES_ENABLE_GPU
                case hercules::proto::MEMORY_GPU: {
        auto status = cuda_memory_manager::alloc(
            (void**)&buffer_, total_byte_size_,
            buffer_attributes_.memory_type_id());
        if (!status.is_ok()) {
          static bool warning_logged = false;
          if (!warning_logged) {
            FLARE_LOG(WARNING) << status
                        << ", falling back to pinned system memory";
            warning_logged = true;
          }

          goto pinned_memory_allocation;
        }
        break;
      }
      pinned_memory_allocation:
#endif  // HERCULES_ENABLE_GPU
                default: {
                    hercules::proto::MemoryType memory_type = buffer_attributes_.memory_type();
                    auto status = pinned_memory_manager::alloc(
                            (void **) &buffer_, total_byte_size_, &memory_type, true);
                    buffer_attributes_.set_memory_type(memory_type);
                    if (!status.is_ok()) {
                        FLARE_LOG(ERROR) << status;
                        buffer_ = nullptr;
                    }
                    break;
                }
            }
        }
        total_byte_size_ = (buffer_ == nullptr) ? 0 : total_byte_size_;
    }

    allocated_memory::~allocated_memory() {
        if (buffer_ != nullptr) {
            switch (buffer_attributes_.memory_type()) {
                case hercules::proto::MEMORY_GPU: {
#ifdef HERCULES_ENABLE_GPU
        auto status = cuda_memory_manager::free(buffer_, buffer_attributes_.memory_type_id());
        if (!status.is_ok()) {
          FLARE_LOG(ERROR) << status;
        }
#endif  // HERCULES_ENABLE_GPU
                    break;
                }

                default: {
                    auto status = pinned_memory_manager::Free(buffer_);
                    if (!status.is_ok()) {
                        FLARE_LOG(ERROR) << status;
                        buffer_ = nullptr;
                    }
                    break;
                }
            }
            buffer_ = nullptr;
        }
    }
}  // namespace hercules::core