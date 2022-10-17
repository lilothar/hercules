
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#ifndef HERCULES_COMMON_SYNC_QUEUE_H_
#define HERCULES_COMMON_SYNC_QUEUE_H_

#include <condition_variable>
#include <deque>
#include <mutex>

namespace hercules::common {

    //
    // C++11 doesn't have a sync queue so we implement a simple one.
    //
    template<typename Item>
    class sync_queue {
    public:
        sync_queue() {}

        bool Empty() {
            std::lock_guard<std::mutex> lk(mu_);
            return queue_.empty();
        }

        Item Get() {
            std::unique_lock<std::mutex> lk(mu_);
            if (queue_.empty()) {
                cv_.wait(lk, [this] { return !queue_.empty(); });
            }
            auto res = std::move(queue_.front());
            queue_.pop_front();
            return res;
        }

        void Put(const Item &value) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                queue_.push_back(value);
            }
            cv_.notify_all();
        }

        void Put(Item &&value) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                queue_.push_back(std::move(value));
            }
            cv_.notify_all();
        }

    private:
        std::mutex mu_;
        std::condition_variable cv_;
        std::deque<Item> queue_;
    };

}  // namespace hercules::common

#endif  // HERCULES_COMMON_SYNC_QUEUE_H_
