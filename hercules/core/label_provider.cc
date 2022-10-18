//
// Created by liyinbin on 2022/10/18.
//

#include "hercules/core/label_provider.h"
#include "hercules/common/error_code.h"
#include "hercules/common/macros.h"
#include <flare/files/sequential_read_file.h>
#include <flare/files/readline_file.h>

namespace hercules::core {


    const std::string & label_provider::get_label(const std::string &name, size_t index) const {
        static const std::string not_found;

        auto itr = label_map_.find(name);
        if (itr == label_map_.end()) {
            return not_found;
        }

        if (itr->second.size() <= index) {
            return not_found;
        }

        return itr->second[index];
    }

    flare::result_status label_provider::add_labels(const std::string &name, const std::string &filepath) {
        std::string label_file_contents;
        flare::sequential_read_file file;
        RETURN_IF_ERROR(file.open(filepath));

        RETURN_IF_ERROR(file.read(&label_file_contents));

        auto p = label_map_.insert(std::make_pair(name, std::vector<std::string>()));
        if (!p.second) {
            return flare::result_status(
                    hercules::proto::ERROR_INTERNAL, "multiple label files for '" + name + "'");
        }

        auto itr = p.first;
        flare::readline_file label_file;
        RETURN_IF_ERROR(label_file.open(label_file_contents));
        std::vector<std::string_view> lines = label_file.lines();
        for (auto& line : lines) {
            itr->second.push_back(std::string(line.data(), line.size()));
        }

        return flare::result_status::success();
    }

    const std::vector<std::string> &label_provider::get_labels(const std::string &name) {
        static const std::vector<std::string> not_found;
        auto itr = label_map_.find(name);
        if (itr == label_map_.end()) {
            return not_found;
        }
        return itr->second;
    }

    flare::result_status label_provider::add_labels(
            const std::string &name, const std::vector<std::string> &labels) {
        label_map_.emplace(name, labels);
        return flare::result_status::success();
    }

}  // namespace hercules::core
