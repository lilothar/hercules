//
// Created by liyinbin on 2022/10/13.
//

#ifndef HERCULES_CORE_PINED_MEMORY_MANAGER_H_
#define HERCULES_CORE_PINED_MEMORY_MANAGER_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <flare/base/result_status.h>
#include "hercules/common/model_config.h"
#include "hercules/proto/memory_type.pb.h"
#include <boost/interprocess/managed_external_buffer.hpp>

namespace hercules::core {
    // This is a singleton class responsible for maintaining pinned memory pool
    // used by the inference server. Pinned memory allocations and deallocations
    // must be requested via functions provided by this class.
    class pinned_memory_manager {
    public:
        // Options to configure pinned memeory manager.
        struct options {
            options(
                    uint64_t b = 0,
                    const hercules::common::two_level_map_config &host_policy_map = {})
                    : pinned_memory_pool_byte_size_(b), host_policy_map_(host_policy_map) {
            }

            uint64_t pinned_memory_pool_byte_size_;
            hercules::common::two_level_map_config host_policy_map_;
        };

        ~pinned_memory_manager();

        // Create the pinned memory manager based on 'options' specified.
        // Return flare::result_status object indicating success or failure.
        static flare::result_status create(const options &options);

        // Allocate pinned memory with the requested 'size' and return the pointer
        // in 'ptr'. If 'allow_nonpinned_fallback' is true, regular system memory
        // will be allocated as fallback in the case where pinned memory fails to
        // be allocated.
        // Return flare::result_status object indicating success or failure.
        static flare::result_status alloc(
                void **ptr, uint64_t size, hercules::proto::MemoryType *allocated_type,
                bool allow_nonpinned_fallback);

        // Free the memory allocated by the pinned memory manager.
        // Return flare::result_status object indicating success or failure.
        static flare::result_status Free(void *ptr);

    protected:
        // Provide explicit control on the lifecycle of the CUDA memory manager,
        // for testing only.
        static void reset();

    private:
        class pinned_memory {
        public:
            pinned_memory(void *pinned_memory_buffer, uint64_t size);

            ~pinned_memory();

            void *pinned_memory_buffer_;
            std::mutex buffer_mtx_;
            boost::interprocess::managed_external_buffer managed_pinned_memory_;
        };

        pinned_memory_manager() = default;

        flare::result_status alloc_internal(
                void **ptr, uint64_t size, hercules::proto::MemoryType *allocated_type,
                bool allow_nonpinned_fallback, pinned_memory *pinned_memory_buffer);

        flare::result_status free_internal(void *ptr);

        void add_pinned_memory_buffer(
                const std::shared_ptr<pinned_memory> &pinned_memory_buffer,
                unsigned long node_mask);

        static std::unique_ptr<pinned_memory_manager> instance_;
        static uint64_t pinned_memory_byte_size_;

        std::mutex info_mtx_;
        std::map<void *, std::pair<bool, pinned_memory *>> memory_info_;
        std::map<unsigned long, std::shared_ptr<pinned_memory>> pinned_memory_buffers_;
    };

}  // namespace hercules::core

#endif  // HERCULES_CORE_PINED_MEMORY_MANAGER_H_
