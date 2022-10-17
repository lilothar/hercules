

/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#ifndef HERCULES_COMMON_THREAD_POOL_H_
#define HERCULES_COMMON_THREAD_POOL_H_

#include <condition_variable>
#include <functional>
#include <queue>
#include <thread>

namespace hercules::common {

    class thread_pool {
    public:
        explicit thread_pool(std::size_t thread_count);

        ~thread_pool();

        thread_pool(const thread_pool &) = delete;

        thread_pool &operator=(const thread_pool &) = delete;

        using func_task = std::function<void(void)>;

        // Assigns "task" to the task queue for a worker thread to execute when
        // available. This will not track the return value of the task.
        void enqueue(func_task &&task);

        // Returns the number of threads in thread pool
        size_t size() { return workers_.size(); }

    private:
        std::queue<func_task> task_queue_;
        std::mutex queue_mtx_;
        std::condition_variable cv_;
        std::vector<std::thread> workers_;
        // If true, tells pool to stop accepting work and tells awake worker threads
        // to exit when no tasks are left on the queue.
        bool stop_ = false;
    };

}  // namespace hercules::common

#endif  // HERCULES_COMMON_THREAD_POOL_H_
