//
// Created by liyinbin on 2022/10/18.
//

#include "hercules/core/infer_response.h"
#include <flare/log/logging.h>

namespace hercules::core {

    flare::result_status inference_response_factory::CreateResponse(
            std::unique_ptr<inference_response> *response) {
        uint64_t response_index = total_response_idx_++;
        response->reset(new inference_response(
                model_, id_, allocator_, alloc_userp_, response_fn_, response_userp_,
                response_delegator_, response_index, request_id_));
#ifdef TRITON_ENABLE_TRACING
        (*response)->SetTrace(trace_);
#endif  // TRITON_ENABLE_TRACING
        return flare::result_status::success();
    }

    flare::result_status
    inference_response_factory::SendFlags(const uint32_t flags) const {
        if (response_delegator_ != nullptr) {
            std::unique_ptr<inference_response> response(
                    new inference_response(response_fn_, response_userp_));
            response_delegator_(std::move(response), flags);
        } else {
            void *userp = response_userp_;
            response_fn_(nullptr /* response */, flags, userp);
        }
        return flare::result_status::success();
    }

    //
    // inference_response
    //
    inference_response::inference_response(const std::shared_ptr<Model> &model, const std::string &id,
                                           const response_allocator *allocator, void *alloc_userp,
                                           inference_response_complete_func response_fn,
                                           void *response_userp,
                                           const std::function<
                                                   void(std::unique_ptr<inference_response> &&,
                                                        const uint32_t)> &delegator,
                                           uint64_t response_idx, uint64_t request_id)
            : model_(model), id_(id), allocator_(allocator), alloc_userp_(alloc_userp),
              response_fn_(response_fn), response_userp_(response_userp),
              response_delegator_(delegator), null_response_(false),
              response_idx_(response_idx), request_id_(request_id) {
        response_start_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count();

        // If the allocator has a start_fn then invoke it.
        TRITONSERVER_ResponseAllocatorStartFn_t start_fn = allocator_->StartFn();
        if (start_fn != nullptr) {
            LOG_TRITONSERVER_ERROR(
                    start_fn(
                            reinterpret_cast<TRITONSERVER_ResponseAllocator *>(
                                    const_cast<response_allocator *>(allocator_)),
                            alloc_userp_),
                    "response allocation start failed");
        }
    }

    inference_response::inference_response(inference_response_complete_func response_fn, void *response_userp)
            : response_fn_(response_fn), response_userp_(response_userp),
              null_response_(true) {
    }

    const std::string &
    inference_response::ModelName() const {
        static const std::string unknown("<unknown>");
        return (model_ == nullptr) ? unknown : model_->Name();
    }

    int64_t
    inference_response::ActualModelVersion() const {
        return (model_ == nullptr) ? -1 : model_->Version();
    }

    flare::result_status
    inference_response::AddParameter(const char *name, const char *value) {
        parameters_.emplace_back(name, value);
        return flare::result_status::success();
    }

    flare::result_status
    inference_response::AddParameter(const char *name, const int64_t value) {
        parameters_.emplace_back(name, value);
        return flare::result_status::success();
    }

    flare::result_status
    inference_response::AddParameter(const char *name, const bool value) {
        parameters_.emplace_back(name, value);
        return flare::result_status::success();
    }

    flare::result_status
    inference_response::AddOutput(
            const std::string &name, const inference::DataType datatype,
            const std::vector<int64_t> &shape, inference_response::Output **output) {
        outputs_.emplace_back(name, datatype, shape, allocator_, alloc_userp_);

        FLARE_LOG(INFO) << "add response output: " << outputs_.back();

        if (model_ != nullptr) {
            const inference::ModelOutput *output_config;
            RETURN_IF_ERROR(model_->GetOutput(name, &output_config));
            if (output_config->has_reshape()) {
                const bool has_batch_dim = (model_->Config().max_batch_size() > 0);
                outputs_.back().Reshape(has_batch_dim, output_config);
            }
        }

        if (output != nullptr) {
            *output = std::addressof(outputs_.back());
        }

        return flare::result_status::success();
    }

    flare::result_status
    inference_response::AddOutput(
            const std::string &name, const inference::DataType datatype,
            std::vector<int64_t> &&shape, inference_response::Output **output) {
        outputs_.emplace_back(
                name, datatype, std::move(shape), allocator_, alloc_userp_);

        LOG_VERBOSE(1) << "add response output: " << outputs_.back();

        if (model_ != nullptr) {
            const inference::ModelOutput *output_config;
            RETURN_IF_ERROR(model_->GetOutput(name, &output_config));
            if (output_config->has_reshape()) {
                const bool has_batch_dim = (model_->Config().max_batch_size() > 0);
                outputs_.back().Reshape(has_batch_dim, output_config);
            }
        }

        if (output != nullptr) {
            *output = std::addressof(outputs_.back());
        }

        return flare::result_status::success();
    }

    flare::result_status
    inference_response::ClassificationLabel(
            const inference_response::Output &output, const uint32_t class_index,
            const char **label) const {
        const auto &label_provider = model_->GetLabelProvider();
        const std::string &l = label_provider->GetLabel(output.Name(), class_index);
        if (l.empty()) {
            *label = nullptr;
        } else {
            *label = l.c_str();
        }

        return flare::result_status::success();
    }

    flare::result_status
    inference_response::Send(
            std::unique_ptr<inference_response> &&response, const uint32_t flags) {
#ifdef TRITON_ENABLE_TRACING
        response->TraceOutputTensors(
      TRITONSERVER_TRACE_TENSOR_BACKEND_OUTPUT, "inference_response Send");
#endif  // TRITON_ENABLE_TRACING

        if (response->response_delegator_ != nullptr) {
            auto ldelegator = std::move(response->response_delegator_);
            ldelegator(std::move(response), flags);
            return flare::result_status::success();
        }
        void *userp = response->response_userp_;
        if (response->null_response_) {
            response->response_fn_(nullptr /* response */, flags, userp);
        } else {
            auto &response_fn = response->response_fn_;
            response_fn(
                    reinterpret_cast<TRITONSERVER_InferenceResponse *>(response.release()),
                    flags, userp);
        }
        return flare::result_status::success();
    }

    flare::result_status
    inference_response::send_with_status(
            std::unique_ptr<inference_response> &&response, const uint32_t flags,
            const flare::result_status &status) {
        response->status_ = status;
        return inference_response::Send(std::move(response), flags);
    }

#ifdef TRITON_ENABLE_TRACING
    flare::result_status
inference_response::TraceOutputTensors(
    TRITONSERVER_InferenceTraceActivity activity, const std::string& msg)
{
  const auto& outputs = this->Outputs();
  uint32_t output_count = outputs.size();

  for (uint32_t idx = 0; idx < output_count; ++idx) {
    const Output& output = outputs[idx];

    // output data
    const char* cname = output.Name().c_str();
    TRITONSERVER_DataType datatype = DataTypeToTriton(output.DType());
    const std::vector<int64_t>& oshape = output.Shape();
    const int64_t* shape = &oshape[0];
    uint64_t dim_count = oshape.size();
    const void* base;
    size_t byte_size;
    hercules::proto::MemoryType memory_type;
    int64_t memory_type_id;
    void* userp;

    flare::result_status status = output.DataBuffer(
        &base, &byte_size, &memory_type, &memory_type_id, &userp);
    if (!status.IsOk()) {
      LOG_STATUS_ERROR(
          status,
          std::string(TRITONSERVER_InferenceTraceActivityString(activity)) +
              ": " + msg + ": fail to get data buffer: " + status.Message());
      return status;
    }

    INFER_TRACE_TENSOR_ACTIVITY(
        this->trace_, activity, cname, datatype, base, byte_size, shape,
        dim_count, memory_type, memory_type_id);
  }

  return flare::result_status::success();
}
#endif  // TRITON_ENABLE_TRACING

//
// inference_response::Output
//
    inference_response::Output::~Output() {
        flare::result_status status = ReleaseDataBuffer();
        if (!status.IsOk()) {
            LOG_ERROR << "failed to release buffer for output '" << name_
                      << "': " << status.AsString();
        }
    }

    void
    inference_response::Output::Reshape(
            const bool has_batch_dim, const inference::ModelOutput *output_config) {
        std::deque<int64_t> variable_size_values;

        const int64_t batch_dim =
                (has_batch_dim && (shape_.size() > 0)) ? shape_[0] : -1;
        const size_t batch_dim_offset = (has_batch_dim) ? 1 : 0;

        const auto &from_shape = output_config->reshape().shape();
        const auto &to_shape = output_config->dims();
        for (int64_t idx = 0; idx < from_shape.size(); idx++) {
            if (from_shape[idx] == -1) {
                variable_size_values.push_back(shape_[idx + batch_dim_offset]);
            }
        }

        shape_.clear();
        if (batch_dim >= 0) {
            shape_.push_back(batch_dim);
        }

        for (const auto &dim : to_shape) {
            if (dim == -1) {
                shape_.push_back(variable_size_values.front());
                variable_size_values.pop_front();
            } else {
                shape_.push_back(dim);
            }
        }
    }

    flare::result_status
    inference_response::Output::DataBuffer(
            const void **buffer, size_t *buffer_byte_size,
            hercules::proto::MemoryType *memory_type, int64_t *memory_type_id,
            void **userp) const {
        *buffer = allocated_buffer_;
        *buffer_byte_size = buffer_attributes_.byte_size();
        *memory_type = buffer_attributes_.memory_type();
        *memory_type_id = buffer_attributes_.memory_type_id();
        *userp = allocated_userp_;
        return flare::result_status::success();
    }

    flare::result_status
    inference_response::Output::AllocateDataBuffer(
            void **buffer, size_t buffer_byte_size,
            hercules::proto::MemoryType *memory_type, int64_t *memory_type_id) {
        if (allocated_buffer_ != nullptr) {
            return flare::result_status(
                    flare::result_status::Code::ALREADY_EXISTS,
                    "allocated buffer for output '" + name_ + "' already exists");
        }

        hercules::proto::MemoryType actual_memory_type = *memory_type;
        int64_t actual_memory_type_id = *memory_type_id;
        void *alloc_buffer_userp = nullptr;

        RETURN_IF_TRITONSERVER_ERROR(allocator_->AllocFn()(
                reinterpret_cast<TRITONSERVER_ResponseAllocator *>(
                        const_cast<ResponseAllocator *>(allocator_)),
                name_.c_str(), buffer_byte_size, *memory_type, *memory_type_id,
                alloc_userp_, buffer, &alloc_buffer_userp, &actual_memory_type,
                &actual_memory_type_id));

        // Only call the buffer attributes API if it is set.
        if (allocator_->BufferAttributesFn() != nullptr) {
            RETURN_IF_TRITONSERVER_ERROR(allocator_->BufferAttributesFn()(
                    reinterpret_cast<TRITONSERVER_ResponseAllocator *>(
                            const_cast<ResponseAllocator *>(allocator_)),
                    name_.c_str(),
                    reinterpret_cast<TRITONSERVER_BufferAttributes *>(&buffer_attributes_),
                    alloc_userp_, alloc_buffer_userp));
        }

        allocated_buffer_ = *buffer;
        buffer_attributes_.SetByteSize(buffer_byte_size);
        buffer_attributes_.SetMemoryType(actual_memory_type);
        buffer_attributes_.SetMemoryTypeId(actual_memory_type_id);

        allocated_userp_ = alloc_buffer_userp;
        *memory_type = actual_memory_type;
        *memory_type_id = actual_memory_type_id;

        return flare::result_status::success();
    }

    flare::result_status
    inference_response::Output::ReleaseDataBuffer() {
        TRITONSERVER_Error *err = nullptr;

        if (allocated_buffer_ != nullptr) {
            err = allocator_->ReleaseFn()(
                    reinterpret_cast<TRITONSERVER_ResponseAllocator *>(
                            const_cast<ResponseAllocator *>(allocator_)),
                    allocated_buffer_, allocated_userp_, buffer_attributes_.ByteSize(),
                    buffer_attributes_.MemoryType(), buffer_attributes_.MemoryTypeId());
        }

        allocated_buffer_ = nullptr;
        buffer_attributes_.set_byte_size(0);
        buffer_attributes_.set_memory_type(TRITONSERVER_MEMORY_CPU);
        buffer_attributes_.set_memory_type_id(0);
        allocated_userp_ = nullptr;

        RETURN_IF_TRITONSERVER_ERROR(err);

        return flare::result_status::success();
    }

    std::ostream &
    operator<<(std::ostream &out, const inference_response &response) {
        out << "[0x" << std::addressof(response) << "] "
            << "response id: " << response.Id() << ", model: " << response.ModelName()
            << ", actual version: " << response.ActualModelVersion() << std::endl;

        out << "status:" << response.response_status() << std::endl;

        out << "outputs:" << std::endl;
        for (const auto &output : response.Outputs()) {
            out << "[0x" << std::addressof(output) << "] " << output << std::endl;
        }

        return out;
    }

    std::ostream &
    operator<<(std::ostream &out, const inference_response::Output &output) {
        out << "output: " << output.Name()
            << ", type: " << triton::common::DataTypeToProtocolString(output.DType())
            << ", shape: " << triton::common::DimsListToString(output.Shape());
        return out;
    }

}