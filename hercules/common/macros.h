//
// Created by liyinbin on 2022/10/17.
//

#ifndef HERCULES_COMMON_MACROS_H_
#define HERCULES_COMMON_MACROS_H_

#include <flare/base/result_status.h>

#define RETURN_IF_ERROR(S)        \
  do {                            \
    const flare::result_status& status__ = (S); \
    if (!status__.is_ok()) {       \
      return status__;            \
    }                             \
  } while (false)

#endif  // HERCULES_COMMON_MACROS_H_
