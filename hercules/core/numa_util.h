//
// Created by liyinbin on 2022/10/13.
//

#ifndef HERCULES_CORE_NUMA_UTIL_H_
#define HERCULES_CORE_NUMA_UTIL_H_

#include <map>
#include <thread>
#include <vector>
#include "flare/base/result_status.h"
#include "hercules/common/model_config.h"

namespace hercules::core {


    // Helper function to set memory policy and thread affinity on current thread
    flare::result_status set_numa_config_on_thread(const hercules::common::map_config& host_policy);

    // Restrict the memory allocation to specific NUMA node.
    flare::result_status set_numa_memory_policy(
            const hercules::common::map_config& host_policy);

    // Retrieve the node mask used to set memory policy for the current thread
    flare::result_status get_numa_memory_policy_node_mask(unsigned long* node_mask);

    // Reset the memory allocation setting.
    flare::result_status reset_numa_memory_policy();

    // Set a thread affinity to be on specific cpus.
    flare::result_status set_numa_thread_affinity(
            std::thread::native_handle_type thread,
            const hercules::common::map_config& host_policy);


}  // namespace hercules::core
#endif  // HERCULES_CORE_NUMA_UTIL_H_
