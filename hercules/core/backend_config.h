
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#ifndef HERCULES_CORE_BACKEND_CONFIG_H_
#define HERCULES_CORE_BACKEND_CONFIG_H_

#include <flare/base/result_status.h>
#include "hercules/common/model_config.h"

namespace hercules::core {


    /// Get a key's string value from a backend configuration.
    flare::result_status backend_configuration(
            const hercules::common::map_config& config, const std::string& key,
            std::string* val);

    /// Convert a backend configuration string  value into a double.
    flare::result_status backend_configuration_parse_string_to_double(
            const std::string& str, double* val);

    /// Convert a backend configuration string  value into a bool.
    flare::result_status backend_configuration_parse_string_to_bool(const std::string& str, bool* val);

    /// Get the global backends directory from the backend configuration.
    flare::result_status backend_configuration_global_backends_directory(
            const hercules::common::two_level_map_config& config_map,
            std::string* dir);

    /// Get the minimum compute capability from the backend configuration.
    flare::result_status backend_configuration_min_compute_capability(
            const hercules::common::two_level_map_config& config_map, double* mcc);

    /// Get the model configuration auto-complete setting from the backend
    /// configuration.
    flare::result_status backend_configuration_auto_complete_config(
            const hercules::common::two_level_map_config& config_map, bool* acc);

    /// Convert a backend name to the specialized version of that name
    /// based on the backend configuration. For example, "tensorflow" will
    /// convert to either "tensorflow1" or "tensorflow2" depending on how
    /// hercules is run.
    flare::result_status backend_configuration_specialize_backend_name(
            const hercules::common::two_level_map_config& config_map,
            const std::string& backend_name, std::string* specialized_name);

    /// Return the shared library name for a backend.
    flare::result_status BackendConfigurationBackendLibraryName(
            const std::string& backend_name, std::string* libname);

    /// Get GPU memory limit fraction for model loading
    /// from the backend configuration.
    flare::result_status backend_configuration_model_load_gpu_fraction(
            const hercules::common::two_level_map_config& config_map,
            const int device_id, double* memory_limit);
}  // namespace hercules::core

#endif  // HERCULES_CORE_BACKEND_CONFIG_H_
