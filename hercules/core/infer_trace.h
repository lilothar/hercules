//
// Created by liyinbin on 2022/10/18.
//

#ifndef HERCULES_INFER_TRACE_H
#define HERCULES_INFER_TRACE_H

#include <atomic>
#include <string>
#include <functional>
#include "hercules/core/inference_trace_level.h"
#include "hercules/core/inference_trace_activity.h"
#include "hercules/proto/data_type.pb.h"
#include "hercules/proto/memory_type.pb.h"

namespace hercules::core {

#define HERCULES_ENABLE_TRACING
#ifdef HERCULES_ENABLE_TRACING

    //
    // InferenceTrace
    //
    // Interface to InferenceTrace to report trace events.
    //

    class InferenceTrace;

    typedef std::function<void(InferenceTrace *, InferenceTraceActivity, uint64_t, void *)> inference_trace_activity_fn;

    typedef std::function<void(InferenceTrace *, InferenceTraceActivity, const char *,
    hercules::proto::DataType, const void *, size_t,  const int64_t *, uint64_t, hercules::proto::MemoryType,int64_t, void* )> inference_trace_tensor_activity_fn;

    typedef std::function<void(InferenceTrace *, void*)> inference_trace_release_fn;
    class InferenceTrace {
    public:
        InferenceTrace(
                const inference_trace_level level, const uint64_t parent_id,
                inference_trace_activity_fn activity_fn,
                inference_trace_tensor_activity_fn tensor_activity_fn,
                inference_trace_release_fn release_fn, void *userp)
                : level_(level), id_(next_id_++), parent_id_(parent_id),
                  activity_fn_(activity_fn), tensor_activity_fn_(tensor_activity_fn),
                  release_fn_(release_fn), userp_(userp) {
        }

        InferenceTrace *SpawnChildTrace();

        int64_t Id() const { return id_; }

        int64_t ParentId() const { return parent_id_; }

        const std::string &ModelName() const { return model_name_; }

        int64_t ModelVersion() const { return model_version_; }

        void SetModelName(const std::string &n) { model_name_ = n; }

        void SetModelVersion(int64_t v) { model_version_ = v; }

        // Report trace activity.
        void Report(
                const InferenceTraceActivity activity, uint64_t timestamp_ns) {
            if ((level_ & TRACE_LEVEL_TIMESTAMPS) > 0) {
                activity_fn_(
                        reinterpret_cast<InferenceTrace *>(this), activity,
                        timestamp_ns, userp_);
            }
        }

        // Report trace activity at the current time.
        void ReportNow(const InferenceTraceActivity activity) {
            if ((level_ & TRACE_LEVEL_TIMESTAMPS) > 0) {
                Report(
                        activity, std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
            }
        }

        // Report tensor trace activity.
        void ReportTensor(
                const InferenceTraceActivity activity, const char *name,
                hercules::proto::DataType datatype, const void *base, size_t byte_size,
                const int64_t *shape, uint64_t dim_count,
                hercules::proto::MemoryType memory_type, int64_t memory_type_id) {
            if ((level_ & TRACE_LEVEL_TENSORS) > 0) {
                tensor_activity_fn_(
                        reinterpret_cast<InferenceTrace *>(this), activity, name,
                        datatype, base, byte_size, shape, dim_count, memory_type,
                        memory_type_id, userp_);
            }
        }

        // Release the trace. Call the trace release callback.
        void Release();

    private:
        const inference_trace_level level_;
        const uint64_t id_;
        const uint64_t parent_id_;

        inference_trace_activity_fn activity_fn_;
        inference_trace_tensor_activity_fn tensor_activity_fn_;
        inference_trace_release_fn release_fn_;
        void *userp_;

        std::string model_name_;
        int64_t model_version_;

        // Maintain next id statically so that trace id is unique even
        // across traces
        static std::atomic<uint64_t> next_id_;
    };

    //
    // InferenceTraceProxy
    //
    // Object attached as shared_ptr to InferenceRequest and
    // InferenceResponse(s) being traced as part of a single inference
    // request.
    //
    class InferenceTraceProxy {
    public:
        InferenceTraceProxy(InferenceTrace *trace) : trace_(trace) {}

        ~InferenceTraceProxy() { trace_->Release(); }

        int64_t Id() const { return trace_->Id(); }

        int64_t ParentId() const { return trace_->ParentId(); }

        const std::string &ModelName() const { return trace_->ModelName(); }

        int64_t ModelVersion() const { return trace_->ModelVersion(); }

        void SetModelName(const std::string &n) { trace_->SetModelName(n); }

        void SetModelVersion(int64_t v) { trace_->SetModelVersion(v); }

        void Report(
                const InferenceTraceActivity activity, uint64_t timestamp_ns) {
            trace_->Report(activity, timestamp_ns);
        }

        void ReportNow(const InferenceTraceActivity activity) {
            trace_->ReportNow(activity);
        }

        void ReportTensor(
                const InferenceTraceActivity activity, const char *name,
                hercules::proto::DataType datatype, const void *base, size_t byte_size,
                const int64_t *shape, uint64_t dim_count,
                hercules::proto::MemoryType memory_type, int64_t memory_type_id) {
            trace_->ReportTensor(
                    activity, name, datatype, base, byte_size, shape, dim_count,
                    memory_type, memory_type_id);
        }

        std::shared_ptr<InferenceTraceProxy> SpawnChildTrace();

    private:
        InferenceTrace *trace_;
    };

#endif  // HERCULES_ENABLE_TRACING

//
// Macros to generate trace activity
//
#ifdef HERCULES_ENABLE_TRACING
#define INFER_TRACE_ACTIVITY(T, A, TS_NS) \
  {                                       \
    const auto& trace = (T);              \
    const auto ts_ns = (TS_NS);           \
    if (trace != nullptr) {               \
      trace->Report(A, ts_ns);            \
    }                                     \
  }
#define INFER_TRACE_ACTIVITY_NOW(T, A) \
  {                                    \
    const auto& trace = (T);           \
    if (trace != nullptr) {            \
      trace->ReportNow(A);             \
    }                                  \
  }
#define INFER_TRACE_TENSOR_ACTIVITY(T, A, N, D, BA, BY, S, DI, MT, MTI) \
  {                                                                     \
    const auto& trace = (T);                                            \
    if (trace != nullptr) {                                             \
      trace->ReportTensor(A, N, D, BA, BY, S, DI, MT, MTI);             \
    }                                                                   \
  }
#else
#define INFER_TRACE_ACTIVITY(T, A, TS_NS)
#define INFER_TRACE_ACTIVITY_NOW(T, A)
#define INFER_TRACE_TENSOR_ACTIVITY(T, A, N, D, BA, BY, S, DI, MT, MTI)
#endif  // HERCULES_ENABLE_TRACING

}  // namespace hercules::core

#endif //HERCULES_INFER_TRACE_H
