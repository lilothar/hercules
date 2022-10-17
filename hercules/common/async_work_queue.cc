
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#include "hercules/common/error_code.h"
#include "hercules/common/async_work_queue.h"

namespace hercules::common {

    async_work_queue::~async_work_queue() {
        get_singleton()->thread_pool_.reset();
    }

    async_work_queue *
    async_work_queue::get_singleton() {
        static async_work_queue singleton;
        return &singleton;
    }

    flare::result_status
    async_work_queue::initialize(size_t worker_count) {
        if (worker_count < 1) {
            return flare::result_status(
                    ERROR_INVALID_ARG,
                    "Async work queue must be initialized with positive 'worker_count'");
        }

        static std::mutex init_mtx;
        std::lock_guard<std::mutex> lk(init_mtx);

        if (get_singleton()->thread_pool_) {
            return flare::result_status(
                    ERROR_ALREADY_EXISTS,
                    "Async work queue has been initialized with " +
                    std::to_string(get_singleton()->thread_pool_->size()) +
                    " 'worker_count'");
        }

        get_singleton()->thread_pool_.reset(new thread_pool(worker_count));
        return flare::result_status::success();
    }

    size_t
    async_work_queue::worker_count() {
        if (!get_singleton()->thread_pool_) {
            return 0;
        }
        return get_singleton()->thread_pool_->size();
    }

    flare::result_status
    async_work_queue::add_task(std::function<void(void)> &&task) {
        if (!get_singleton()->thread_pool_) {
            return flare::result_status(
                    ERROR_UNAVAILABLE,
                    "Async work queue must be initialized before adding task");
        }
        get_singleton()->thread_pool_->enqueue(std::move(task));

        return flare::result_status::success();
    }

    void
    async_work_queue::reset() {
        // Reconstruct the singleton to reset it
        get_singleton()->~async_work_queue();
        new(get_singleton()) async_work_queue();
    }

}  // namespace hercules::common
