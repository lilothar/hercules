//
// Created by liyinbin on 2022/10/18.
//

#include "hercules/core/infer_trace.h"

namespace hercules::core {


#ifdef HERCULES_ENABLE_TRACING

    // Start the trace id at 1, because id 0 is reserved to indicate no
    // parent.
    std::atomic<uint64_t> InferenceTrace::next_id_(1);

    InferenceTrace *InferenceTrace::SpawnChildTrace() {
        InferenceTrace *trace = new InferenceTrace(
                level_, id_, activity_fn_, tensor_activity_fn_, release_fn_, userp_);
        return trace;
    }

    void
    InferenceTrace::Release() {
        release_fn_(reinterpret_cast<InferenceTrace *>(this), userp_);
    }

    std::shared_ptr<InferenceTraceProxy>
    InferenceTraceProxy::SpawnChildTrace() {
        std::shared_ptr<InferenceTraceProxy> strace_proxy =
                std::make_shared<InferenceTraceProxy>(trace_->SpawnChildTrace());
        return strace_proxy;
    }

#endif  // HERCULES_ENABLE_TRACING
}  // namespace hercules::core