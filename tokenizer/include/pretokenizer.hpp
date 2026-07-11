#pragma once
#include <string>
#include <vector>

namespace bbpe {

class Pretokenizer {
public:
    static std::vector<std::string> split(const std::string& text);
};

} // namespace bbpe
