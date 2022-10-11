
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "hercules/core/data_type.h"

namespace hercules::core {
    std::string_view to_string_view(const hercules::proto::DataType &type) {
        switch (type) {
            case hercules::proto::TYPE_BOOL:
                return "BOOL";
            case hercules::proto::TYPE_UINT8:
                return "UINT8";
            case hercules::proto::TYPE_UINT16:
                return "UINT16";
            case hercules::proto::TYPE_UINT32:
                return "UINT32";
            case hercules::proto::TYPE_UINT64:
                return "UINT64";
            case hercules::proto::TYPE_INT8:
                return "INT8";
            case hercules::proto::TYPE_INT16:
                return "INT16";
            case hercules::proto::TYPE_INT32:
                return "INT32";
            case hercules::proto::TYPE_INT64:
                return "INT64";
            case hercules::proto::TYPE_FP16:
                return "FP16";
            case hercules::proto::TYPE_FP32:
                return "FP32";
            case hercules::proto::TYPE_FP64:
                return "FP64";
            case hercules::proto::TYPE_BYTES:
                return "BYTES";
            case hercules::proto::TYPE_BF16:
                return "BF16";
            default:
                break;
        }

        return "<invalid>";
    }

    hercules::proto::DataType string_to_data_type(const std::string_view &sv) {
        auto dtype = sv.data();
        auto len = sv.size();

        if (sv.size() < 4 || sv.size() > 6) {
            return hercules::proto::DataType::TYPE_INVALID;
        }


        if ((*dtype == 'I') && (len != 6)) {
            if ((dtype[1] == 'N') && (dtype[2] == 'T')) {
                if ((dtype[3] == '8') && (len == 4)) {
                    return hercules::proto::DataType::TYPE_INT8;
                } else if ((dtype[3] == '1') && (dtype[4] == '6')) {
                    return hercules::proto::DataType::TYPE_INT16;
                } else if ((dtype[3] == '3') && (dtype[4] == '2')) {
                    return hercules::proto::DataType::TYPE_INT32;
                } else if ((dtype[3] == '6') && (dtype[4] == '4')) {
                    return hercules::proto::DataType::TYPE_INT64;
                }
            }
        } else if ((*dtype == 'U') && (len != 4)) {
            if ((dtype[1] == 'I') && (dtype[2] == 'N') && (dtype[3] == 'T')) {
                if ((dtype[4] == '8') && (len == 5)) {
                    return hercules::proto::DataType::TYPE_UINT8;
                } else if ((dtype[4] == '1') && (dtype[5] == '6')) {
                    return hercules::proto::DataType::TYPE_UINT16;
                } else if ((dtype[4] == '3') && (dtype[5] == '2')) {
                    return hercules::proto::DataType::TYPE_UINT32;
                } else if ((dtype[4] == '6') && (dtype[5] == '4')) {
                    return hercules::proto::DataType::TYPE_UINT64;
                }
            }
        } else if ((*dtype == 'F') && (dtype[1] == 'P') && (len == 4)) {
            if ((dtype[2] == '1') && (dtype[3] == '6')) {
                return hercules::proto::DataType::TYPE_FP16;
            } else if ((dtype[2] == '3') && (dtype[3] == '2')) {
                return hercules::proto::DataType::TYPE_FP32;
            } else if ((dtype[2] == '6') && (dtype[3] == '4')) {
                return hercules::proto::DataType::TYPE_FP64;
            }
        } else if (*dtype == 'B') {
            switch (dtype[1]) {
                case 'Y':
                    if (!strcmp(dtype + 2, "TES")) {
                        return hercules::proto::DataType::TYPE_BYTES;
                    }
                    break;
                case 'O':
                    if (!strcmp(dtype + 2, "OL")) {
                        return hercules::proto::DataType::TYPE_BOOL;
                    }
                    break;
                case 'F':
                    if (!strcmp(dtype + 2, "16")) {
                        return hercules::proto::DataType::TYPE_BF16;
                    }
                    break;
            }
        }

        return hercules::proto::DataType::TYPE_INVALID;
    }

    uint32_t data_type_size(hercules::proto::DataType type) {
        switch (type) {
            case hercules::proto::TYPE_BOOL:
            case hercules::proto::TYPE_INT8:
            case hercules::proto::TYPE_UINT8:
                return 1;
            case hercules::proto::TYPE_INT16:
            case hercules::proto::TYPE_UINT16:
            case hercules::proto::TYPE_FP16:
            case hercules::proto::TYPE_BF16:
                return 2;
            case hercules::proto::TYPE_INT32:
            case hercules::proto::TYPE_UINT32:
            case hercules::proto::TYPE_FP32:
                return 4;
            case hercules::proto::TYPE_INT64:
            case hercules::proto::TYPE_UINT64:
            case hercules::proto::TYPE_FP64:
                return 8;
            case hercules::proto::TYPE_BYTES:
                return 0;
            default:
                break;
        }

        return 0;
    }
}  // namespace hercules::core