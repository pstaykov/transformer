#pragma once
#include <string>
#include <vector>
#include <optional>
#include <functional>
#include <cstdint>
#include <filesystem>

namespace bbpe {

std::optional<std::string> extract_json_text_field(
    const std::string& line,
    const std::string& field);

class CorpusReader {
public:
    explicit CorpusReader(std::vector<std::string> paths,
                          std::string text_field = "text",
                          size_t min_chars = 1);

    uint64_t estimate_total_bytes() const;
    const std::vector<std::filesystem::path>& files() const { return files_; }

    void for_each_document(
        const std::function<void(const std::string&)>& callback) const;

private:
    std::vector<std::filesystem::path> files_;
    std::string text_field_;
    size_t min_chars_;

    void stream_file(
        const std::filesystem::path& path,
        const std::function<void(const std::string&)>& callback) const;
};

} // namespace bbpe
