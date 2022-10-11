
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#ifndef HERCULES_CORE_DATA_TYPE_H_
#define HERCULES_CORE_DATA_TYPE_H_

#include <string_view>
#include <string>
#include "hercules/proto/data_type.pb.h"

namespace hercule::core {

    /// get the string  representation of the data type, the result is a string_view,
    /// we don't copy it as a string to reduce the memory copy.

    [[nodiscard]] std::string_view to_string_view(const hercules::proto::DataType &type);

    /// get the data type of a corresponding string.
    [[nodiscard]] hercules::proto::DataType string_to_data_type(const std::string_view &sv);

    [[nodiscard]] uint32_t data_type_size(hercules::proto::DataType type);
}  // namespace hercule::core
#endif  // HERCULES_CORE_DATA_TYPE_H_
