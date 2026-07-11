#include "corpus_reader.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace bbpe {

std::optional<std::string> extract_json_text_field(
    const std::string& line, const std::string& field) {

    const std::string key = "\"" + field + "\"";
    size_t kp = line.find(key);
    if (kp == std::string::npos) return std::nullopt;
    size_t col = line.find(':', kp + key.size());
    if (col == std::string::npos) return std::nullopt;
    size_t qs = line.find('"', col);
    if (qs == std::string::npos) return std::nullopt;
    ++qs;

    std::string out;
    out.reserve(128);
    for (size_t i = qs; i < line.size() && line[i] != '"'; ) {
        if (line[i] == '\\' && i + 1 < line.size()) {
            switch (line[i+1]) {
                case 'n':  out += '\n'; break;
                case 't':  out += '\t'; break;
                case 'r':  out += '\r'; break;
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'u':
                    if (i+5 < line.size()) {
                        unsigned cp = std::stoul(line.substr(i+2,4), nullptr, 16);
                        if      (cp < 0x80)  { out += static_cast<char>(cp); }
                        else if (cp < 0x800) {
                            out += static_cast<char>(0xC0|(cp>>6));
                            out += static_cast<char>(0x80|(cp&0x3F));
                        } else {
                            out += static_cast<char>(0xE0|(cp>>12));
                            out += static_cast<char>(0x80|((cp>>6)&0x3F));
                            out += static_cast<char>(0x80|(cp&0x3F));
                        }
                        i += 4;
                    }
                    break;
                default: out += line[i+1];
            }
            i += 2;
        } else {
            out += line[i++];
        }
    }
    return out;
}

CorpusReader::CorpusReader(std::vector<std::string> paths,
                           std::string text_field,
                           size_t min_chars)
    : text_field_(std::move(text_field)), min_chars_(min_chars) {

    for (const auto& p : paths) {
        fs::path fp(p);
        if (fs::is_directory(fp)) {
            for (const auto& e : fs::recursive_directory_iterator(fp)) {
                if (!e.is_regular_file()) continue;
                auto name = e.path().string();
                // C++17-kompatibler Suffix-Check (ends_with ist erst C++20)
                bool is_txt = name.size() >= 4 &&
                              name.compare(name.size()-4, 4, ".txt") == 0;
                if (is_txt || name.find(".jsonl") != std::string::npos)
                    files_.push_back(e.path());
            }
        } else if (fs::is_regular_file(fp)) {
            files_.push_back(fp);
        } else {
            std::cerr << "[WARN] Pfad nicht gefunden: " << p << "\n";
        }
    }

    if (files_.empty())
        throw std::runtime_error("Keine .txt/.jsonl-Dateien gefunden.");

    std::cerr << "Gefundene Dateien: " << files_.size() << "\n";
}

uint64_t CorpusReader::estimate_total_bytes() const {
    uint64_t total = 0;
    for (const auto& f : files_) {
        std::error_code ec;
        auto sz = fs::file_size(f, ec);
        if (!ec) total += sz;
    }
    return total;
}

void CorpusReader::stream_file(
    const fs::path& path,
    const std::function<void(const std::string&)>& cb) const {

    // WICHTIG: Kein manueller pubsetbuf()-Aufruf mehr. Der Standard-Puffer
    // von std::ifstream ist ausreichend performant und funktioniert
    // zuverlässig -- der vorherige manuelle Puffer-Hack führte dazu,
    // dass das Programm beim Lesen hing.
    std::ifstream in(path);
    if (!in) {
        std::cerr << "[WARN] Kann nicht öffnen: " << path << "\n";
        return;
    }

    const bool is_jsonl = path.string().find(".jsonl") != std::string::npos;
    std::string line;
    line.reserve(4096);

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (is_jsonl) {
            auto text = extract_json_text_field(line, text_field_);
            if (text && text->size() >= min_chars_) cb(*text);
        } else {
            if (line.size() >= min_chars_) cb(line);
        }
    }
}

void CorpusReader::for_each_document(
    const std::function<void(const std::string&)>& cb) const {
    for (const auto& f : files_) stream_file(f, cb);
}

} // namespace bbpe
