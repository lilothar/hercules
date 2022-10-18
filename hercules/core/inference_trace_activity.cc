//
// Created by liyinbin on 2022/10/18.
//

#include "hercules/core/inference_trace_activity.h"

namespace hercules::core {
    std::string_view to_string_view(InferenceTraceActivity activity) {
        switch (activity) {
            case TRACE_REQUEST_START:
                return "REQUEST_START";
            case TRACE_QUEUE_START:
                return "QUEUE_START";
            case TRACE_COMPUTE_START:
                return "COMPUTE_START";
            case TRACE_COMPUTE_INPUT_END:
                return "COMPUTE_INPUT_END";
            case TRACE_COMPUTE_OUTPUT_START:
                return "COMPUTE_OUTPUT_START";
            case TRACE_COMPUTE_END:
                return "COMPUTE_END";
            case TRACE_REQUEST_END:
                return "REQUEST_END";
            case TRACE_TENSOR_QUEUE_INPUT:
                return "TENSOR_QUEUE_INPUT";
            case TRACE_TENSOR_BACKEND_INPUT:
                return "TENSOR_BACKEND_INPUT";
            case TRACE_TENSOR_BACKEND_OUTPUT:
                return "TENSOR_BACKEND_OUTPUT";
        }

        return "<unknown>";
    }
}