//
// Created by liyinbin on 2022/10/12.
//

#ifndef HERCULES_CORE_MEMORY_BASE_H_
#define HERCULES_CORE_MEMORY_BASE_H_

#include <cstddef>
#include <cstdint>
#include "hercules/core/buffer_attributes.h"
#include "flare/base/profile.h"

namespace hercules::core {

    class memory_base {
    public:
        // Get the 'idx'-th data block in the buffer. Using index to avoid
        // maintaining internal state such that one buffer can be shared
        // across multiple providers.
        // 'idx' zero base index. Valid indices are continuous.
        // 'byte_size' returns the byte size of the chunk of bytes.
        // 'memory_type' returns the memory type of the chunk of bytes.
        // 'memory_type_id' returns the memory type id of the chunk of bytes.
        // Return the pointer to the data block. Returns nullptr if 'idx' is
        // out of range
        virtual const char* buffer_at(
                size_t idx, size_t* byte_size, hercules::proto::MemoryType* memory_type,
                int64_t* memory_type_id) const = 0;

        // Similar to the above buffer_at but with buffer_attributes.
        virtual const char* buffer_at(
                size_t idx, buffer_attributes** buffer_attributes) = 0;

        // Get the number of contiguous buffers composing the memory.
        size_t buffer_count() const { return buffer_count_; }

        // Return the total byte size of the data buffer
        size_t total_byte_size() const { return total_byte_size_; }

    protected:
        memory_base() : total_byte_size_(0), buffer_count_(0) {}
        size_t total_byte_size_;
        size_t buffer_count_;
    };

    class memory_reference : public memory_base {
    public:
        // Create a read-only data buffer as a reference to other data buffer
        memory_reference();

        //\see memory_base::buffer_at()
        const char* buffer_at(
                size_t idx, size_t* byte_size, hercules::proto::MemoryType* memory_type,
                int64_t* memory_type_id) const override;

        const char* buffer_at(
                size_t idx, buffer_attributes** buffer_attributes) override;

        // Add a 'buffer' with 'byte_size' as part of this data buffer
        // Return the index of the buffer
        size_t add_buffer(
                const char* buffer, size_t byte_size, hercules::proto::MemoryType memory_type,
                int64_t memory_type_id);

        size_t add_buffer(const char* buffer, buffer_attributes* buffer_attributes);

        // Add a 'buffer' with 'byte_size' as part of this data buffer in the front
        // Return the index of the buffer
        size_t add_buffer_front(
                const char* buffer, size_t byte_size, hercules::proto::MemoryType memory_type,
                int64_t memory_type_id);

    private:
        struct Block {
            Block(
                    const char* buffer, size_t byte_size,
                    hercules::proto::MemoryType memory_type, int64_t memory_type_id)
                    : buffer_(buffer), buffer_attributes_(buffer_attributes(
                    byte_size, memory_type, memory_type_id, nullptr))
            {
            }

            Block(const char* buffer, buffer_attributes* buffer_attributes)
                    : buffer_(buffer), buffer_attributes_(*buffer_attributes)
            {
            }
            const char* buffer_;
            buffer_attributes buffer_attributes_;
        };
        std::vector<Block> buffer_;
    };

    class mutable_memory : public memory_base {
    public:
        // Create a mutable data buffer referencing to other data buffer.
        mutable_memory(
                char* buffer, size_t byte_size,  hercules::proto::MemoryType memory_type,
                int64_t memory_type_id);

        virtual ~mutable_memory() {}

        //\see Memory::buffer_at()
        const char* buffer_at(
                size_t idx, size_t* byte_size,  hercules::proto::MemoryType* memory_type,
                int64_t* memory_type_id) const override;

        //\see Memory::buffer_at()
        const char* buffer_at(
                size_t idx, buffer_attributes** buffer_attributes) override;

        // Return a pointer to the base address of the mutable buffer. If
        // non-null 'memory_type' returns the memory type of the chunk of
        // bytes. If non-null 'memory_type_id' returns the memory type id of
        // the chunk of bytes.
        char* mutable_buffer(
                 hercules::proto::MemoryType* memory_type = nullptr,
                int64_t* memory_type_id = nullptr);

        FLARE_DISALLOW_COPY_AND_ASSIGN(mutable_memory);

    protected:
        mutable_memory() : memory_base() {}

        char* buffer_;
        buffer_attributes buffer_attributes_;
    };

    class allocated_memory : public mutable_memory {
    public:
        // Create a continuous data buffer with 'byte_size', 'memory_type' and
        // 'memory_type_id'. Note that the buffer may be created on different memeory
        // type and memory type id if the original request type and id can not be
        // satisfied, thus the function caller should always check the actual memory
        // type and memory type id before use.
        allocated_memory(
                size_t byte_size,  hercules::proto::MemoryType memory_type,
                int64_t memory_type_id);

        ~allocated_memory() override;
    };

}  // namespace hercules::core

#endif // HERCULES_CORE_MEMORY_BASE_H_
