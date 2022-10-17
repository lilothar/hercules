
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#ifndef HERCULES_CORE_INFERENCE_PARAMETER_H_
#define HERCULES_CORE_INFERENCE_PARAMETER_H_

#include <iostream>
#include <string>
#include "hercules/core/parameter_type.h"

namespace hercules::core {

    class inference_parameter {
    public:
        inference_parameter(const char *name, const char *value)
                : name_(name), type_(hercules::proto::PARAMETER_STRING), value_string_(value) {
            byte_size_ = value_string_.size();
        }

        inference_parameter(const char *name, const int64_t value)
                : name_(name), type_(hercules::proto::PARAMETER_INT), value_int64_(value),
                  byte_size_(sizeof(int64_t)) {
        }

        inference_parameter(const char *name, const bool value)
                : name_(name), type_(hercules::proto::PARAMETER_BOOL), value_bool_(value),
                  byte_size_(sizeof(bool)) {
        }

        inference_parameter(const char *name, const void *ptr, const uint64_t size)
                : name_(name), type_(hercules::proto::PARAMETER_BYTES), value_bytes_(ptr),
                  byte_size_(size) {
        }

        // The name of the parameter.
        const std::string &name() const { return name_; }

        // Data type of the parameter.
        hercules::proto::ParameterType type() const { return type_; }

        // Return a pointer to the parameter, or a pointer to the data content
        // if type_ is hercules::proto::PARAMETER_BYTES. This returned pointer must be
        // cast correctly based on 'type_'.
        //   hercules::proto::PARAMETER_STRING -> const char*
        //   hercules::proto::PARAMETER_INT -> int64_t*
        //   hercules::proto::PARAMETER_BOOL -> bool*
        //   hercules::proto::PARAMETER_BYTES -> const void*
        const void *value_pointer() const;

        // Return the data byte size of the parameter.
        uint64_t value_byte_size() const { return byte_size_; }

        // Return the parameter value string, the return value is valid only if
        // Type() returns hercules::proto::PARAMETER_STRING
        const std::string &value_string() const { return value_string_; }

    private:
        friend std::ostream &operator<<(
                std::ostream &out, const inference_parameter &parameter);

        std::string name_;
        hercules::proto::ParameterType type_;

        std::string value_string_;
        int64_t value_int64_;
        bool value_bool_;
        const void *value_bytes_;
        uint64_t byte_size_;
    };

    std::ostream &operator<<(
            std::ostream &out, const inference_parameter &parameter);
}  // namespace hercules::core

#endif  // HERCULES_CORE_INFERENCE_PARAMETER_H_
