//
// Created by liyinbin on 2022/10/18.
//

#include "hercules/core/inference_trace_level.h"

namespace hercules::core {

    std::string_view to_string_view(inference_trace_level level) {
        switch (level) {
            case TRACE_LEVEL_DISABLED:
                return "DISABLED";
            case TRACE_LEVEL_TIMESTAMPS:
                return "TIMESTAMPS";
            case TRACE_LEVEL_TENSORS:
                return "TENSORS";
        }
        return "<unknown>";
    }
}  // namespace hercules::core
