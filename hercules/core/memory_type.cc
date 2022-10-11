
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "hercules/core/memory_type.h"

namespace hercules::core {

    std::string_view to_string_view(hercules::proto::MemoryType type) {
        switch (type) {
            case hercules::proto::MEMORY_CPU:
                return "CPU";
            case hercules::proto::MEMORY_CPU_BINDING:
                return "CPU_PINNED";
            case hercules::proto::MEMORY_GPU:
                return "GPU";
            default:
                break;
        }

        return "<invalid>";
    }
}  // namespace hercules::core
