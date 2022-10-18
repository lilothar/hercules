//
// Created by liyinbin on 2022/10/18.
//

#ifndef HERCULES_CORE_INFERENCE_TRACE_ACTIVITY_H_
#define HERCULES_CORE_INFERENCE_TRACE_ACTIVITY_H_

#include <string_view>

namespace hercules::core {
    enum InferenceTraceActivity {
        TRACE_REQUEST_START = 0,
        TRACE_QUEUE_START = 1,
        TRACE_COMPUTE_START = 2,
        TRACE_COMPUTE_INPUT_END = 3,
        TRACE_COMPUTE_OUTPUT_START = 4,
        TRACE_COMPUTE_END = 5,
        TRACE_REQUEST_END = 6,
        TRACE_TENSOR_QUEUE_INPUT = 7,
        TRACE_TENSOR_BACKEND_INPUT = 8,
        TRACE_TENSOR_BACKEND_OUTPUT = 9
    };

    std::string_view to_string_view(InferenceTraceActivity activity);
}  // namespace hercules::core

#endif  // HERCULES_CORE_INFERENCE_TRACE_ACTIVITY_H_
