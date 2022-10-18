//
// Created by liyinbin on 2022/10/18.
//

#ifndef HERCULES_CORE_LABEL_PROVIDER_H_
#define HERCULES_CORE_LABEL_PROVIDER_H_

#include <string>
#include <cstddef>
#include <unordered_map>
#include <flare/base/profile.h>
#include <flare/base/result_status.h>

namespace hercules::core {

    // Provides classification labels.
    class label_provider {
    public:
        label_provider() = default;

        // Return the label associated with 'name' for a given
        // 'index'. Return empty string if no label is available.
        const std::string& get_label(const std::string& name, size_t index) const;

        // Associate with 'name' a set of labels initialized from a given
        // 'filepath'. Within the file each label is specified on its own
        // line. The first label (line 0) is the index-0 label, the second
        // label (line 1) is the index-1 label, etc.
        flare::result_status add_labels(const std::string& name, const std::string& filepath);

        // Return the labels associated with 'name'. Return empty vector if no labels
        // are available.
        const std::vector<std::string>& get_labels(const std::string& name);

        // Associate with 'name' a set of 'labels'
        flare::result_status add_labels(
                const std::string& name, const std::vector<std::string>& labels);

    private:
        FLARE_DISALLOW_COPY_AND_ASSIGN(label_provider);

        std::unordered_map<std::string, std::vector<std::string>> label_map_;
    };

}
#endif  // HERCULES_CORE_LABEL_PROVIDER_H_
