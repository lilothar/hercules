
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "hercules/core/parameter_type.h"

namespace hercules::core {
    std::string_view to_string_view(hercules::proto::ParameterType type) {
        switch (type) {
            case hercules::proto::PARAMETER_STRING:
                return "STRING";
            case hercules::proto::PARAMETER_INT:
                return "INT";
            case hercules::proto::PARAMETER_BOOL:
                return "BOOL";
            default:
                break;
        }

        return "<invalid>";
    }
}  // namespace hercules::core