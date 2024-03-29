
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#ifndef HERCULES_CORE_PARAMETER_TYPE_H_
#define HERCULES_CORE_PARAMETER_TYPE_H_

#include <string_view>
#include "hercules/proto/parameter_type.pb.h"

namespace hercules::core {

    [[nodiscard]] std::string_view to_string_view(hercules::proto::ParameterType type);
}  // namespace hercules::core

#endif  // HERCULES_CORE_PARAMETER_TYPE_H_
