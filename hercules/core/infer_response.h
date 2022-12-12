//
// Created by liyinbin on 2022/10/18.
//

#ifndef HERCULES_CORE_INFER_RESPONSE_H_
#define HERCULES_CORE_INFER_RESPONSE_H_

#include <cstdint>
#include <cstddef>
#include <functional>
#include <string>
#include <deque>
#include <flare/base/result_status.h>
#include "hercules/core/response_allocator.h"
#include "hercules/core/memory_type.h"
#include "hercules/core/data_type.h"
#include "hercules/core/inference_parameter.h"
#include "hercules/core/buffer_attributes.h"
#include "hercules/proto/model_config.pb.h"

namespace hercules::core {

    class Model;
    class inference_response;
    typedef std::function<void(inference_response*, const int, void*)> inference_response_complete_func;
    //
    // An inference response factory.
    //
    class inference_response_factory {
    public:
        inference_response_factory() = default;

        inference_response_factory(
                const std::shared_ptr<Model>& model, const std::string& id,
                const response_allocator* allocator, void* alloc_userp,
                inference_response_complete_func response_fn,
                void* response_userp,
                const std::function<void(
                std::unique_ptr<inference_response>&&, const uint32_t)>& delegator,
        const uint64_t request_id)
        : model_(model), id_(id), allocator_(allocator),
                alloc_userp_(alloc_userp), response_fn_(response_fn),
                response_userp_(response_userp), response_delegator_(delegator),
                request_id_(request_id), total_response_idx_(0)
        {
        }

        const response_allocator* Allocator() { return allocator_; }
        void* AllocatorUserp() { return alloc_userp_; }

        flare::result_status SetResponseDelegator(
                const std::function<void(
                std::unique_ptr<inference_response>&&, const uint32_t)>& delegator)
        {
            response_delegator_ = delegator;
            return flare::result_status::success();
        }

        // Create a new response.
        flare::result_status CreateResponse(std::unique_ptr<inference_response>* response);

        // Send a "null" response with 'flags'.
        flare::result_status SendFlags(const uint32_t flags) const;

#ifdef TRITON_ENABLE_TRACING
        const std::shared_ptr<InferenceTraceProxy>& Trace() const { return trace_; }
  void SetTrace(const std::shared_ptr<InferenceTraceProxy>& trace)
  {
    trace_ = trace;
  }
  void ReleaseTrace() { trace_ = nullptr; }
#endif  // TRITON_ENABLE_TRACING

    private:
        // The model associated with this factory. For normal
        // requests/responses this will always be defined and acts to keep
        // the model loaded as long as this factory is live. It may be
        // nullptr for cases where the model itself created the request
        // (like running requests for warmup) and so must protect any uses
        // to handle the nullptr case.
        std::shared_ptr<Model> model_;

        // The ID of the corresponding request that should be included in every
        // response. This is a property that can be optionally provided by the user.
        std::string id_;

        // The response allocator and user pointer. The 'allocator_' is a
        // raw pointer because it is owned by the client, and the client is
        // responsible for ensuring that the lifetime of the allocator
        // extends longer that any request or response that depend on the
        // allocator.

        const response_allocator* allocator_;
        void* alloc_userp_;

        // The response callback function and user pointer.
        inference_response_complete_func response_fn_;
        void* response_userp_;

        // Delegator to be invoked on sending responses.
        std::function<void(std::unique_ptr<inference_response>&&, const uint32_t)>
        response_delegator_;


#ifdef TRITON_ENABLE_TRACING
        // Inference trace associated with this response.
        std::shared_ptr<InferenceTraceProxy> trace_;
#endif  // TRITON_ENABLE_TRACING

        // The internal unique ID of the request. This is a unique identification
        // that triton attaches to each request in case user does not provide id_.
        uint64_t request_id_;

        // Response index
        std::atomic<uint64_t> total_response_idx_;
    };

    //
    // An inference response.
    //

    class inference_response {
    public:
        // Output tensor
        class Output {
        public:
            Output(
                    const std::string& name, const hercules::proto::DataType datatype,
                    const std::vector<int64_t>& shape, const response_allocator* allocator,
                    void* alloc_userp)
                    : name_(name), datatype_(datatype), shape_(shape),
                      allocator_(allocator), alloc_userp_(alloc_userp),
                      allocated_buffer_(nullptr)
            {
            }
            Output(
                    const std::string& name, const hercules::proto::DataType datatype,
                    std::vector<int64_t>&& shape, const response_allocator* allocator,
                    void* alloc_userp)
                    : name_(name), datatype_(datatype), shape_(std::move(shape)),
                      allocator_(allocator), alloc_userp_(alloc_userp),
                      allocated_buffer_(nullptr)
            {
            }

            ~Output();

            // The name of the output tensor.
            const std::string& Name() const { return name_; }

            // Data type of the output tensor.
            hercules::proto::DataType DType() const { return datatype_; }

            // The shape of the output tensor.
            const std::vector<int64_t>& Shape() const { return shape_; }

            buffer_attributes* GetBufferAttributes() { return &buffer_attributes_; }

            // Reshape the output tensor. This function must only be called
            // for outputs that have respace specified in the model
            // configuration.
            void Reshape(
                    const bool has_batch_dim, const hercules::proto::ModelOutput* output_config);

            // Get information about the buffer allocated for this output
            // tensor's data. If no buffer is allocated 'buffer' will return
            // nullptr and the other returned values will be undefined.
            flare::result_status DataBuffer(
                    const void** buffer, size_t* buffer_byte_size,
                    hercules::proto::MemoryType* memory_type, int64_t* memory_type_id,
                    void** userp) const;

            // Allocate the buffer that should be used for this output
            // tensor's data. 'buffer' must return a buffer of size
            // 'buffer_byte_size'.  'memory_type' acts as both input and
            // output. On input gives the buffer memory type preferred by the
            // caller and on return holds the actual memory type of
            // 'buffer'. 'memory_type_id' acts as both input and output. On
            // input gives the buffer memory type id preferred by the caller
            // and returns the actual memory type id of 'buffer'. Only a
            // single buffer may be allocated for the output at any time, so
            // multiple calls to AllocateDataBuffer without intervening
            // ReleaseDataBuffer call will result in an error.
            flare::result_status AllocateDataBuffer(
                    void** buffer, const size_t buffer_byte_size,
                    hercules::proto::MemoryType* memory_type, int64_t* memory_type_id);

            // Release the buffer that was previously allocated by
            // AllocateDataBuffer(). Do nothing if AllocateDataBuffer() has
            // not been called.
            flare::result_status ReleaseDataBuffer();

        private:
            FLARE_DISALLOW_COPY_AND_ASSIGN(Output);
            friend std::ostream& operator<<(
                    std::ostream& out, const inference_response::Output& output);

            std::string name_;
            hercules::proto::DataType datatype_;
            std::vector<int64_t> shape_;

            // The response allocator and user pointer.
            const response_allocator* allocator_;
            void* alloc_userp_;

            // Information about the buffer allocated by
            // AllocateDataBuffer(). This information is needed by
            // DataBuffer() and ReleaseDataBuffer().
            void* allocated_buffer_;
            buffer_attributes buffer_attributes_;
            void* allocated_userp_;
        };

        // inference_response
        inference_response(
                const std::shared_ptr<Model>& model, const std::string& id,
                const response_allocator* allocator, void* alloc_userp,
                inference_response_complete_func response_fn,
                void* response_userp,
                const std::function<void(
                std::unique_ptr<inference_response>&&, const uint32_t)>& delegator,
        const uint64_t response_idx, const uint64_t request_id);

        // "null" inference_response is a special instance of inference_response which
        // contains minimal information for calling inference_response::Send,
        // inference_response::NullResponse. nullptr will be passed as response in
        // 'response_fn'.
        inference_response(inference_response_complete_func response_fn, void* response_userp);

        const std::string& Id() const { return id_; }
        const std::string& ModelName() const;
        int64_t ActualModelVersion() const;
        const flare::result_status& response_status() const { return status_; }

        // The response parameters.
        [[nodiscard]] const std::deque<inference_parameter>& Parameters() const
        {
            return parameters_;
        }

        // Add an parameter to the response.
        flare::result_status AddParameter(const char* name, const char* value);
        flare::result_status AddParameter(const char* name, const int64_t value);
        flare::result_status AddParameter(const char* name, const bool value);

        // The response outputs.
        [[nodiscard]] const std::deque<Output>& Outputs() const { return outputs_; }

        // The response index.
        uint64_t ResponseIdx() const { return response_idx_; }

        // The unique request ID stored in triton.
        uint64_t RequestUniqueId() const { return request_id_; }
        // The timestamp in nanoseconds when the response started.
        uint64_t ResponseStartNs() const { return response_start_; }

        // Add an output to the response. If 'output' is non-null
        // return a pointer to the newly added output.
        flare::result_status AddOutput(
                const std::string& name, const hercules::proto::DataType datatype,
                const std::vector<int64_t>& shape, Output** output = nullptr);
        flare::result_status AddOutput(
                const std::string& name, const hercules::proto::DataType datatype,
                std::vector<int64_t>&& shape, Output** output = nullptr);

        // Get the classification label associated with an output. Return
        // 'label' == nullptr if no label.
        flare::result_status ClassificationLabel(
                const Output& output, const uint32_t class_index,
                const char** label) const;

        // Send the response with success status. Calling this function
        // releases ownership of the response object and gives it to the
        // callback function.
        static flare::result_status Send(
                std::unique_ptr<inference_response>&& response, const uint32_t flags);

        // Send the response with explicit status. Calling this function
        // releases ownership of the response object and gives it to the
        // callback function.
        static flare::result_status send_with_status(
                std::unique_ptr<inference_response>&& response, const uint32_t flags,
                const flare::result_status& status);

#ifdef TRITON_ENABLE_TRACING
        const std::shared_ptr<InferenceTraceProxy>& Trace() const { return trace_; }
  void SetTrace(const std::shared_ptr<InferenceTraceProxy>& trace)
  {
    trace_ = trace;
  }
  void ReleaseTrace() { trace_ = nullptr; }
#endif  // TRITON_ENABLE_TRACING

    private:
        FLARE_DISALLOW_COPY_AND_ASSIGN(inference_response);
        friend std::ostream& operator<<(
                std::ostream& out, const inference_response& response);

#ifdef TRITON_ENABLE_TRACING
        flare::result_status TraceOutputTensors(
        TRITONSERVER_InferenceTraceActivity activity, const std::string& msg);
#endif  // TRITON_ENABLE_TRACING

        // The model associated with this factory. For normal
        // requests/responses this will always be defined and acts to keep
        // the model loaded as long as this factory is live. It may be
        // nullptr for cases where the model itself created the request
        // (like running requests for warmup) and so must protect any uses
        // to handle the nullptr case.
        std::shared_ptr<Model> model_;

        // The ID of the corresponding request that should be included in
        // every response.
        std::string id_;

        // Error status for the response.
        flare::result_status status_;

        // The parameters of the response. Use a deque so that there is no
        // reallocation.
        std::deque<inference_parameter> parameters_;

        // The result tensors. Use a deque so that there is no reallocation.
        std::deque<Output> outputs_;

        // The response allocator and user pointer.
        const response_allocator* allocator_;
        void* alloc_userp_;

        // The response callback function and user pointer.
        inference_response_complete_func response_fn_;
        void* response_userp_;

        // Delegator to be invoked on sending responses.
        std::function<void(std::unique_ptr<inference_response>&&, const uint32_t)>
        response_delegator_;

        bool null_response_;

        // A static variable counting the total number of responses created for the
        // request.
        static uint64_t total_responses_count;

#ifdef TRITON_ENABLE_TRACING
        // Inference trace associated with this response.
        std::shared_ptr<InferenceTraceProxy> trace_;
#endif  // TRITON_ENABLE_TRACING

        // Representing the response index.
        uint64_t response_idx_;

        // Representing the request id that the response was created from.
        uint64_t request_id_;
        // Timestamp in nanoseconds when the response started.
        uint64_t response_start_;
    };

    std::ostream& operator<<(std::ostream& out, const inference_response& response);
    std::ostream& operator<<(
            std::ostream& out, const inference_response::Output& output);

}  // namespace hercules::core

#endif  // HERCULES_CORE_INFER_RESPONSE_H_
