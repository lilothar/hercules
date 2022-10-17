
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#ifndef HERCULES_CORE_BUFFER_ATTRIBUTES_H_
#define HERCULES_CORE_BUFFER_ATTRIBUTES_H_

#include <iterator>
#include <vector>
#include <cstddef>
#include <cstdint>
#include "hercules/core/memory_type.h"

namespace hercules::core {

    class buffer_attributes {
    public:
        buffer_attributes(
                size_t byte_size, hercules::proto::MemoryType memory_type,
                int64_t memory_type_id, char cuda_ipc_handle[64]);

        buffer_attributes() {
            memory_type_ = hercules::proto::MEMORY_CPU;
            memory_type_id_ = 0;
            cuda_ipc_handle_.reserve(64);
        }

        // Set the buffer byte size
        void set_byte_size(const size_t &byte_size);

        // Set the buffer memory_type
        void set_memory_type(const hercules::proto::MemoryType &memory_type);

        // Set the buffer memory type id
        void set_memory_type_id(const int64_t &memory_type_id);

        // Set the cuda ipc handle
        void set_cuda_ipc_handle(void *cuda_ipc_handle);

        // Get the cuda ipc handle
        void *cuda_ipc_handle();

        // Get the byte size
        size_t byte_size() const;

        // Get the memory type
        hercules::proto::MemoryType memory_type() const;

        // Get the memory type id
        int64_t memory_type_id() const;

    private:
        size_t byte_size_;
        hercules::proto::MemoryType memory_type_;
        int64_t memory_type_id_;
        std::vector<char> cuda_ipc_handle_;
    };
}  // namespace hercules::core

#endif  // HERCULES_CORE_BUFFER_ATTRIBUTES_H_
