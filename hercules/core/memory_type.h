
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#ifndef HERCULES_CORE_MEMORY_TYPE_H_
#define HERCULES_CORE_MEMORY_TYPE_H_

#include <string_view>
#include "hercules/proto/memory_type.pb.h"

namespace hercule::core {

    [[nodiscard]] std::string_view to_string_view(hercules::proto::MemoryType type);
}  // namespace hercule::core

#endif  // HERCULES_CORE_MEMORY_TYPE_H_
