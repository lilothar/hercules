//
// Created by liyinbin on 2022/10/18.
//

#ifndef HERCULES_CORE_INFERENCE_TRACE_LEVEL_H_
#define HERCULES_CORE_INFERENCE_TRACE_LEVEL_H_

#include <string_view>

namespace hercules::core {

    enum inference_trace_level {
        /// Tracing disabled. No trace activities are reported.
        TRACE_LEVEL_DISABLED = 0,
        /// Record timestamps for the inference request.
        TRACE_LEVEL_TIMESTAMPS = 0x4,
        /// Record input and output tensor values for the inference request.
        TRACE_LEVEL_TENSORS = 0x8
    };

    std::string_view to_string_view(inference_trace_level level);
}  // namespace hercules::core
#endif  // HERCULES_CORE_INFERENCE_TRACE_LEVEL_H_
