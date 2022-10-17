
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#ifndef HERCULES_COMMON_ASYNC_WORK_QUEUE_H_
#define HERCULES_COMMON_ASYNC_WORK_QUEUE_H_

#include "hercules/common/thread_pool.h"
#include <flare/base/result_status.h>

namespace hercules::common {

    // Manager for asynchronous worker threads. Use to accelerate copies and
    // other such operations by running them in parallel.
    // Call Initialize to start the worker threads (once) and AddTask to tasks to
    // the queue.

    class async_work_queue {
    public:
        // Start 'worker_count' number of worker threads.
        static flare::result_status initialize(size_t worker_count);

        // Get the number of worker threads.
        static size_t worker_count();

        // Add a 'task' to the queue. The function will take ownership of 'task'.
        // Therefore std::move should be used when calling AddTask.
        static flare::result_status add_task(std::function<void(void)> &&task);

    protected:
        static void reset();

    private:
        async_work_queue() = default;

        ~async_work_queue();

        static async_work_queue *get_singleton();

        std::unique_ptr<thread_pool> thread_pool_;
    };

}  // namespace hercules::common

#endif  // HERCULES_COMMON_ASYNC_WORK_QUEUE_H_
