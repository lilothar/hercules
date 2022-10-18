//
// Created by liyinbin on 2022/10/13.
//

#include "hercules/core/numa_util.h"
#include "hercules/common/error_code.h"
#include "hercules/common/macros.h"
#include <flare/base/profile.h>
#include <flare/log/logging.h>

namespace hercules::core {


    namespace {
        std::string
        VectorToString(const std::vector<int> &vec) {
            std::string str("[");
            for (const auto &element : vec) {
                str += std::to_string(element);
                str += ",";
            }

            str += "]";
            return str;
        }

        flare::result_status
        ParseIntOption(const std::string &msg, const std::string &arg, int *value) {
            try {
                *value = std::stoi(arg);
            }
            catch (const std::invalid_argument &ia) {
                return flare::result_status(hercules::common::ERROR_INVALID_ARG,
                                            msg + ": Can't parse '" + arg + "' to integer");
            }
            return flare::result_status::success();
        }

    }  // namespace

// NUMA setting will be ignored on Windows platform
#ifdef FLARE_PLATFORM_OSX

    flare::result_status
    set_numa_config_on_thread(const hercules::common::map_config &host_policy) {
        return flare::result_status::success();
    }

    flare::result_status
    set_numa_memory_policy(const hercules::common::map_config &host_policy) {
        return flare::result_status::success();
    }

    flare::result_status
    get_numa_memory_policy_node_mask(unsigned long *node_mask) {
        *node_mask = 0;
        return flare::result_status::success();
    }

    flare::result_status
    reset_numa_memory_policy() {
        return flare::result_status::success();
    }

    flare::result_status
    set_numa_thread_affinity(
            std::thread::native_handle_type thread,
            const hercules::common::map_config &host_policy) {
        return flare::result_status::success();
    }

#else
    // Use variable to make sure no NUMA related function is actually called
    // if hercules is not running with NUMA awareness. i.e. Extra docker permission
    // is needed to call the NUMA functions and this ensures backward compatibility.
    thread_local bool numa_set = false;

    flare::result_status
    set_numa_config_on_thread(const std::map<std::string, std::string>&host_policy) {
        // Set thread affinity
        RETURN_IF_ERROR(set_numa_thread_affinity(pthread_self(), host_policy));

        // Set memory policy
        RETURN_IF_ERROR(set_numa_memory_policy(host_policy));

        return flare::result_status::success();
    }

    flare::result_status
    set_numa_memory_policy(const hercules::common::map_config &host_policy) {
        const auto it = host_policy.find("numa-node");
        if (it != host_policy.end()) {
            int node_id;
            RETURN_IF_ERROR(
                    ParseIntOption("Parsing 'numa-node' value", it->second, &node_id));
            FLARE_LOG(INFO) << "Thread is binding to NUMA node " << it->second
                           << ". Max NUMA node count: " << (numa_max_node() + 1);
            numa_set = true;
            unsigned long node_mask = 1UL << node_id;
            if (set_mempolicy(MPOL_BIND, &node_mask, (numa_max_node() + 1) + 1) != 0) {
                return flare::result_status(
                        ERROR_INTERNAL,
                        std::string("Unable to set NUMA memory policy: ") + strerror(errno));
            }
        }
        return flare::result_status::success();
    }

    flare::result_status
    get_numa_memory_policy_node_mask(unsigned long *node_mask) {
        *node_mask = 0;
        int mode;
        if (numa_set &&
            get_mempolicy(&mode, node_mask, numa_max_node() + 1, NULL, 0) != 0) {
            return flare::result_status(
                    ERROR_INTERNAL,
                    std::string("Unable to get NUMA node for current thread: ") +
                    strerror(errno));
        }
        return flare::result_status::success();
    }

    flare::result_status
    reset_numa_memory_policy() {
        if (numa_set && (set_mempolicy(MPOL_DEFAULT, nullptr, 0) != 0)) {
            return flare::result_status(
                    ERROR_INTERNAL,
                    std::string("Unable to reset NUMA memory policy: ") + strerror(errno));
        }
        numa_set = false;
        return flare::result_status::success();
    }

    flare::result_status
    set_numa_thread_affinity(
            std::thread::native_handle_type thread,
            const hercules::common::map_config &host_policy) {
        const auto it = host_policy.find("cpu-cores");
        if (it != host_policy.end()) {
            // Parse CPUs
            std::vector<int> cpus;
            {
                const auto &cpu_str = it->second;
                auto delim_cpus = cpu_str.find(",");
                int current_pos = 0;
                while (true) {
                    auto delim_range = cpu_str.find("-", current_pos);
                    if (delim_range == std::string::npos) {
                        return flare::result_status(
                                ERROR_INVALID_ARG,
                                std::string("host policy setting 'cpu-cores' format is "
                                            "'<lower_cpu_core_id>-<upper_cpu_core_id>'. Got ") +
                                cpu_str.substr(
                                        current_pos, ((delim_cpus == std::string::npos)
                                                      ? (cpu_str.length() + 1)
                                                      : delim_cpus) -
                                                     current_pos));
                    }
                    int lower, upper;
                    RETURN_IF_ERROR(ParseIntOption(
                            "Parsing 'cpu-cores' value",
                            cpu_str.substr(current_pos, delim_range - current_pos), &lower));
                    RETURN_IF_ERROR(ParseIntOption(
                            "Parsing 'cpu-cores' value",
                            (delim_cpus == std::string::npos)
                            ? cpu_str.substr(delim_range + 1)
                            : cpu_str.substr(
                                    delim_range + 1, delim_cpus - (delim_range + 1)),
                            &upper));
                    for (; lower <= upper; ++lower) {
                        cpus.push_back(lower);
                    }
                    // break if the processed range is the last specified range
                    if (delim_cpus != std::string::npos) {
                        current_pos = delim_cpus + 1;
                        delim_cpus = cpu_str.find(",", current_pos);
                    } else {
                        break;
                    }
                }
            }

            FLARE_LOG(INFO)<< "Thread is binding to one of the CPUs: "
                           << VectorToString(cpus);
            numa_set = true;
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            for (int cpu : cpus) {
                CPU_SET(cpu, &cpuset);
            }
            if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset) != 0) {
                return flare::result_status(
                        ERROR_INTERNAL,
                        std::string("Unable to set NUMA thread affinity: ") +
                        strerror(errno));
            }
        }
        return flare::result_status::success();
    }

#endif

}  // namespace hercules::core