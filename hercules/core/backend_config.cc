
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "hercules/core/backend_config.h"
#include "hercules/common/macros.h"
#include "hercules/common/error_code.h"
#include <flare/log/logging.h>

namespace hercules::core {

    namespace {

        flare::result_status get_tf_specialized_backend_name(const hercules::common::two_level_map_config &config_map,
                                                         std::string *specialized_name) {
            std::string tf_version_str = "2";
            const auto &itr = config_map.find("tensorflow");
            if (itr != config_map.end()) {
                if (backend_configuration(itr->second, "version", &tf_version_str).is_ok()) {
                    if ((tf_version_str != "1") && (tf_version_str != "2")) {
                        return flare::result_status(
                                hercules::common::ERROR_INVALID_ARG,
                                "unexpected TensorFlow library version '" + tf_version_str +
                                "', expects 1 or 2.");
                    }
                }
            }

            *specialized_name += tf_version_str;

            return flare::result_status::success();
        }
    }  // namespace

    flare::result_status backend_configuration(
            const hercules::common::map_config &config, const std::string &key,
            std::string *val) {
        for (const auto &pr : config) {
            if (pr.first == key) {
                *val = pr.second;
                return flare::result_status::success();
            }
        }

        return flare::result_status(
                hercules::common::ERROR_INTERNAL,
                std::string("unable to find common backend configuration for '") + key +
                "'");
    }

    flare::result_status
    backend_configuration_parse_string_to_double(const std::string &str, double *val) {
        try {
            *val = std::stod(str);
        }
        catch (...) {
            return flare::result_status(
                    hercules::common::ERROR_INTERNAL,
                    "unable to parse common backend configuration as double");
        }

        return flare::result_status::success();
    }

    flare::result_status
    backend_configuration_parse_string_to_bool(const std::string &str, bool *val) {
        try {
            std::string lowercase_str{str};
            std::transform(
                    lowercase_str.begin(), lowercase_str.end(), lowercase_str.begin(),
                    [](unsigned char c) { return std::tolower(c); });
            *val = (lowercase_str == "true");
        }
        catch (...) {
            return flare::result_status(
                    hercules::common::ERROR_INTERNAL,
                    "unable to parse common backend configuration as bool");
        }

        return flare::result_status::success();
    }

    flare::result_status
    backend_configuration_global_backends_directory(
            const hercules::common::two_level_map_config &config_map, std::string *dir) {
        const auto &itr = config_map.find(std::string());
        if (itr == config_map.end()) {
            return flare::result_status(
                    hercules::common::ERROR_INTERNAL,
                    "unable to find global backends directory configuration");
        }

        RETURN_IF_ERROR(backend_configuration(itr->second, "backend-directory", dir));

        return flare::result_status::success();
    }

    flare::result_status
    backend_configuration_min_compute_capability(
            const hercules::common::two_level_map_config &config_map, double *mcc) {
#ifdef HERCULES_ENABLE_GPU
        *mcc = HERCULES_MIN_COMPUTE_CAPABILITY;
#else
        *mcc = 0;
#endif  // HERCULES_ENABLE_GPU

        const auto &itr = config_map.find(std::string());
        if (itr == config_map.end()) {
            return flare::result_status(
                    hercules::common::ERROR_INTERNAL, "unable to find common backend configuration");
        }

        std::string min_compute_capability_str;
        RETURN_IF_ERROR(backend_configuration(
                itr->second, "min-compute-capability", &min_compute_capability_str));
        RETURN_IF_ERROR(backend_configuration_parse_string_to_double(min_compute_capability_str, mcc));

        return flare::result_status::success();
    }

    flare::result_status
    backend_configuration_auto_complete_config(
            const hercules::common::two_level_map_config &config_map, bool *acc) {
        const auto &itr = config_map.find(std::string());
        if (itr == config_map.end()) {
            return flare::result_status(
                    hercules::common::ERROR_INTERNAL, "unable to find auto-complete configuration");
        }

        std::string auto_complete_config_str;
        RETURN_IF_ERROR(backend_configuration(
                itr->second, "auto-complete-config", &auto_complete_config_str));
        RETURN_IF_ERROR(
                backend_configuration_parse_string_to_bool(auto_complete_config_str, acc));

        return flare::result_status::success();
    }

    flare::result_status
    backend_configuration_specialize_backend_name(
            const hercules::common::two_level_map_config &config_map,
            const std::string &backend_name, std::string *specialized_name) {
        *specialized_name = backend_name;
        if (backend_name == "tensorflow") {
            RETURN_IF_ERROR(get_tf_specialized_backend_name(config_map, specialized_name));
        }

        return flare::result_status::success();
    }

    flare::result_status
    backend_configuration_backend_library_name(
            const std::string &backend_name, std::string *libname) {
#ifdef FLARE_PLATFORM_WINDOWS
        *libname = "hercules_" + backend_name + ".dll";
#elif defined(FLARE_PLATFORM_LINUX)
        *libname = "libhercules_" + backend_name + ".so";
#elif defined(FLARE_PLATFORM_OSX)
        *libname = "libhercules_" + backend_name + ".dylib";
#else
#error "Unsupported backend"
#endif

        return flare::result_status::success();
    }

    flare::result_status
    BackendConfigurationModelLoadGpuFraction(
            const hercules::common::two_level_map_config &config_map,
            const int device_id, double *memory_limit) {
        *memory_limit = 1.0;
        const auto &itr = config_map.find(std::string());
        if (itr == config_map.end()) {
            return flare::result_status(
                    hercules::common::ERROR_INTERNAL,
                    "unable to find global backends directory configuration");
        }

        static std::string key_prefix = "model-load-gpu-limit-device-";
        std::string memory_limit_str;
        auto status = backend_configuration(
                itr->second, key_prefix + std::to_string(device_id), &memory_limit_str);
        // Allow missing key, default to 1.0 (no limit) if the limit is not specified
        if (status.is_ok()) {
            RETURN_IF_ERROR(backend_configuration_parse_string_to_double(
                    memory_limit_str, memory_limit));
        }

        return flare::result_status::success();
    }

}  // namespace hercules::core