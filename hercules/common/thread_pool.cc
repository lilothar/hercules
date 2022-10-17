
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#include "hercules/common/thread_pool.h"
#include <stdexcept>

namespace hercules::common {

    thread_pool::thread_pool(size_t thread_count) {
        if (!thread_count) {
            throw std::invalid_argument("Thread count must be greater than zero.");
        }

        // Define infinite loop for each thread to wait for a task to complete
        const auto worker_loop = [this]() {
            while (true) {
                func_task task;
                {
                    std::unique_lock<std::mutex> lk(queue_mtx_);
                    // Wake if there's a task to do, or the pool has been stopped.
                    cv_.wait(lk, [&]() { return !task_queue_.empty() || stop_; });
                    // Exit condition
                    if (stop_ && task_queue_.empty()) {
                        break;
                    }
                    task = std::move(task_queue_.front());
                    task_queue_.pop();
                }

                // Execute task - ensure function has a valid target
                if (task) {
                    task();
                }
            }
        };

        workers_.reserve(thread_count);
        for (size_t i = 0; i < thread_count; ++i) {
            workers_.emplace_back(worker_loop);
        }
    }

    thread_pool::~thread_pool() {
        {
            std::lock_guard<std::mutex> lk(queue_mtx_);
            // Signal to each worker that it should exit loop when tasks are finished
            stop_ = true;
        }
        // Wake all threads to clean up
        cv_.notify_all();
        for (auto &t : workers_) {
            t.join();
        }
    }

    void
    thread_pool::enqueue(func_task &&task) {
        {
            std::lock_guard<std::mutex> lk(queue_mtx_);
            // Don't accept more work if pool is shutting down
            if (stop_) {
                return;
            }
            task_queue_.push(std::move(task));
        }
        // Only wake one thread per task
        cv_.notify_one();
    }

}  // namespace hercules::common
