
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#include "hercules/core/inference_parameter.h"

namespace hercules::core {

    const void *
    inference_parameter::value_pointer() const {
        switch (type_) {
            case hercules::proto::PARAMETER_STRING:
                return reinterpret_cast<const void *>(value_string_.c_str());
            case hercules::proto::PARAMETER_INT:
                return reinterpret_cast<const void *>(&value_int64_);
            case hercules::proto::PARAMETER_BOOL:
                return reinterpret_cast<const void *>(&value_bool_);
            case hercules::proto::PARAMETER_BYTES:
                return reinterpret_cast<const void *>(value_bytes_);
            default:
                break;
        }

        return nullptr;
    }

    std::ostream &
    operator<<(std::ostream &out, const inference_parameter &parameter) {
        out << "[0x" << std::addressof(parameter) << "] "
            << "name: " << parameter.name()
            << ", type: " << to_string_view(parameter.type())
            << ", value: ";
        return out;
    }
}  // namespace hercules::core
