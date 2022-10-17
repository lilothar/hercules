//
// Created by liyinbin on 2022/10/17.
//

#ifndef HERCULES_CORE_ERROR_CODE_H_
#define HERCULES_CORE_ERROR_CODE_H_

#include "hercules/proto/error_code.pb.h"

namespace hercules::common {

    static const int ERROR_SUCCESS = static_cast<int>(hercules::proto::ERROR_SUCCESS);
    static const int ERROR_UNKNOWN = static_cast<int>(hercules::proto::ERROR_UNKNOWN);
    static const int ERROR_INTERNAL = static_cast<int>(hercules::proto::ERROR_INTERNAL);
    static const int ERROR_NOT_FOUND = static_cast<int>(hercules::proto::ERROR_NOT_FOUND);
    static const int ERROR_INVALID_ARG = static_cast<int>(hercules::proto::ERROR_INVALID_ARG);
    static const int ERROR_UNAVAILABLE = static_cast<int>(hercules::proto::ERROR_UNAVAILABLE);
    static const int ERROR_UNSUPPORTED = static_cast<int>(hercules::proto::ERROR_UNSUPPORTED);
    static const int ERROR_ALREADY_EXISTS = static_cast<int>(hercules::proto::ERROR_ALREADY_EXISTS);

}  // namespace hercules::common

#endif  // HERCULES_CORE_ERROR_CODE_H_
