#pragma once
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef WITH_BBPE
#include "bpe_model.hpp"  // bbpe::Tokenizer
#endif

// Host-side data loading for the CUDA trainer: the tokenizer abstraction
// (byte-level fallback or the real bbpe tokenizer), a small JSON parser, and
// conversation-format (SFT) dataset construction. This mirrors train.py's
// ByteTokenizer / bbpe path and utils/chat.py so `--data-format chat` and
// `--tokenizer bbpe` behave the same as the Python trainer.

// ----------------------------------------------------------------------------
// Tokenizer
// ----------------------------------------------------------------------------
struct DataTokenizer {
    bool is_bbpe = false;
    int vsize = 257;  // byte tokenizer: 256 byte values + 1 pad id
#ifdef WITH_BBPE
    std::shared_ptr<bbpe::Tokenizer> bpe;
#endif

    std::vector<int> encode(const std::string& s) const {
        if (is_bbpe) {
#ifdef WITH_BBPE
            std::vector<bbpe::TokenId> ids = bpe->encode(s);
            return std::vector<int>(ids.begin(), ids.end());
#else
            (void)s;
            return {};
#endif
        }
        std::vector<int> out;
        out.reserve(s.size());
        for (unsigned char ch : s) out.push_back((int)ch);
        return out;
    }

    std::string decode(const std::vector<int>& ids) const {
        if (is_bbpe) {
#ifdef WITH_BBPE
            std::vector<bbpe::TokenId> tid(ids.begin(), ids.end());
            return bpe->decode(tid);
#else
            return "";
#endif
        }
        std::string out;
        out.reserve(ids.size());
        for (int id : ids) if (id >= 0 && id < 256) out.push_back((char)(unsigned char)id);
        return out;
    }

    int vocab_size() const { return vsize; }
};

// Build a tokenizer. `requested_vocab` is used only for bbpe (matches
// train.py's --vocab-size, which sets the model's output dim); it's bumped up
// if the loaded tokenizer actually has more tokens, so no id can index out of
// the embedding table.
inline DataTokenizer make_tokenizer(const std::string& type, const std::string& path,
                                    int requested_vocab) {
    DataTokenizer t;
    if (type == "bbpe") {
#ifdef WITH_BBPE
        t.is_bbpe = true;
        t.bpe = std::make_shared<bbpe::Tokenizer>(bbpe::Tokenizer::load_binary(path));
        int total = (int)t.bpe->vocab_size_total();
        t.vsize = std::max(requested_vocab, total);
        if (t.vsize != requested_vocab)
            fprintf(stderr, "[tokenizer] bumped vocab_size %d -> %d to fit tokenizer's %d tokens\n",
                    requested_vocab, t.vsize, total);
#else
        fprintf(stderr,
                "[tokenizer] this binary was built without bbpe support (WITH_BBPE=OFF); "
                "rebuild with -DWITH_BBPE=ON or use --tokenizer byte\n");
        exit(EXIT_FAILURE);
#endif
    } else {
        t.is_bbpe = false;
        t.vsize = 257;
    }
    return t;
}

// ----------------------------------------------------------------------------
// Minimal JSON parser (objects, arrays, strings w/ escapes, numbers, bools,
// null) - enough for OpenAI-style conversation files. Throws on malformed input.
// ----------------------------------------------------------------------------
namespace minijson {

struct JsonValue {
    enum class Type { Null, Bool, Num, Str, Arr, Obj };
    Type type = Type::Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<JsonValue> arr;
    std::vector<std::pair<std::string, JsonValue>> obj;

    bool is_obj() const { return type == Type::Obj; }
    bool is_arr() const { return type == Type::Arr; }
    bool is_str() const { return type == Type::Str; }

    const JsonValue* get(const std::string& key) const {
        if (type != Type::Obj) return nullptr;
        for (const auto& kv : obj)
            if (kv.first == key) return &kv.second;
        return nullptr;
    }
    bool has(const std::string& key) const { return get(key) != nullptr; }
};

class Parser {
public:
    explicit Parser(const std::string& s) : s_(s) {}

    JsonValue parse() {
        ws();
        JsonValue v = value();
        ws();
        if (i_ != s_.size()) err("trailing characters after JSON value");
        return v;
    }

private:
    const std::string& s_;
    size_t i_ = 0;

    [[noreturn]] void err(const std::string& msg) const {
        throw std::runtime_error("JSON parse error at offset " + std::to_string(i_) + ": " + msg);
    }

    void ws() {
        while (i_ < s_.size()) {
            char c = s_[i_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') i_++;
            else break;
        }
    }

    static void append_utf8(std::string& out, unsigned int cp) {
        if (cp <= 0x7F) {
            out.push_back((char)cp);
        } else if (cp <= 0x7FF) {
            out.push_back((char)(0xC0 | (cp >> 6)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        } else if (cp <= 0xFFFF) {
            out.push_back((char)(0xE0 | (cp >> 12)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        } else {
            out.push_back((char)(0xF0 | (cp >> 18)));
            out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        }
    }

    unsigned int hex4() {
        if (i_ + 4 > s_.size()) err("truncated \\u escape");
        unsigned int v = 0;
        for (int k = 0; k < 4; ++k) {
            char c = s_[i_++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
            else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
            else err("bad hex digit in \\u escape");
        }
        return v;
    }

    std::string parse_string() {
        // caller guarantees s_[i_] == '"'
        i_++;
        std::string out;
        while (i_ < s_.size()) {
            char c = s_[i_++];
            if (c == '"') return out;
            if (c == '\\') {
                if (i_ >= s_.size()) err("truncated escape");
                char e = s_[i_++];
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'n': out.push_back('\n'); break;
                    case 't': out.push_back('\t'); break;
                    case 'r': out.push_back('\r'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'u': {
                        unsigned int cp = hex4();
                        if (cp >= 0xD800 && cp <= 0xDBFF && i_ + 1 < s_.size() &&
                            s_[i_] == '\\' && s_[i_ + 1] == 'u') {
                            i_ += 2;
                            unsigned int lo = hex4();
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        }
                        append_utf8(out, cp);
                        break;
                    }
                    default: err("unknown escape character");
                }
            } else {
                out.push_back(c);
            }
        }
        err("unterminated string");
    }

    JsonValue value() {
        ws();
        if (i_ >= s_.size()) err("unexpected end of input");
        char c = s_[i_];
        if (c == '{') return object();
        if (c == '[') return array();
        if (c == '"') {
            JsonValue v;
            v.type = JsonValue::Type::Str;
            v.str = parse_string();
            return v;
        }
        if (c == 't' || c == 'f') return boolean();
        if (c == 'n') {
            if (s_.compare(i_, 4, "null") != 0) err("bad literal");
            i_ += 4;
            return JsonValue{};
        }
        return number();
    }

    JsonValue boolean() {
        JsonValue v;
        v.type = JsonValue::Type::Bool;
        if (s_.compare(i_, 4, "true") == 0) { v.b = true; i_ += 4; }
        else if (s_.compare(i_, 5, "false") == 0) { v.b = false; i_ += 5; }
        else err("bad boolean literal");
        return v;
    }

    JsonValue number() {
        size_t start = i_;
        if (i_ < s_.size() && (s_[i_] == '-' || s_[i_] == '+')) i_++;
        while (i_ < s_.size()) {
            char c = s_[i_];
            if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') i_++;
            else break;
        }
        if (i_ == start) err("invalid value");
        JsonValue v;
        v.type = JsonValue::Type::Num;
        v.num = strtod(s_.substr(start, i_ - start).c_str(), nullptr);
        return v;
    }

    JsonValue array() {
        JsonValue v;
        v.type = JsonValue::Type::Arr;
        i_++;  // consume '['
        ws();
        if (i_ < s_.size() && s_[i_] == ']') { i_++; return v; }
        while (true) {
            v.arr.push_back(value());
            ws();
            if (i_ >= s_.size()) err("unterminated array");
            if (s_[i_] == ',') { i_++; continue; }
            if (s_[i_] == ']') { i_++; break; }
            err("expected ',' or ']' in array");
        }
        return v;
    }

    JsonValue object() {
        JsonValue v;
        v.type = JsonValue::Type::Obj;
        i_++;  // consume '{'
        ws();
        if (i_ < s_.size() && s_[i_] == '}') { i_++; return v; }
        while (true) {
            ws();
            if (i_ >= s_.size() || s_[i_] != '"') err("expected string key in object");
            std::string key = parse_string();
            ws();
            if (i_ >= s_.size() || s_[i_] != ':') err("expected ':' after object key");
            i_++;
            JsonValue val = value();
            v.obj.emplace_back(std::move(key), std::move(val));
            ws();
            if (i_ >= s_.size()) err("unterminated object");
            if (s_[i_] == ',') { i_++; continue; }
            if (s_[i_] == '}') { i_++; break; }
            err("expected ',' or '}' in object");
        }
        return v;
    }
};

inline JsonValue parse(const std::string& text) { return Parser(text).parse(); }

}  // namespace minijson

// ----------------------------------------------------------------------------
// Conversation-format (SFT) datasets - mirrors utils/chat.py
// ----------------------------------------------------------------------------
struct Message {
    std::string role;
    std::string content;
};
using Conversation = std::vector<Message>;

// A tokenized corpus: token ids plus a per-token predict mask. mask[i]==1 means
// ids[i] is a valid next-token prediction target (all-true for plain text; only
// assistant-turn content for chat data).
struct Dataset {
    std::vector<int> ids;
    std::vector<char> mask;
};

namespace data_detail {

inline std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "[data] could not open %s\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Same as read_file, but reads at most max_bytes (0 = unlimited). Seeks/reads
// directly into a single right-sized buffer instead of slurping the whole
// file via rdbuf() and truncating after the fact - important for corpora
// that are a large fraction of (or bigger than) system RAM, where even a
// transient full-file copy can OOM the box.
inline std::string read_file_capped(const std::string& path, size_t max_bytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "[data] could not open %s\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    f.seekg(0, std::ios::end);
    std::streamoff file_size = f.tellg();
    if (file_size < 0) {
        fprintf(stderr, "[data] could not determine size of %s\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    size_t read_size = (size_t)file_size;
    if (max_bytes > 0 && max_bytes < read_size) read_size = max_bytes;
    f.seekg(0, std::ios::beg);

    std::string buf(read_size, '\0');
    f.read(&buf[0], (std::streamsize)read_size);
    if ((size_t)f.gcount() != read_size) {
        fprintf(stderr, "[data] short read on %s (%zu of %zu bytes)\n", path.c_str(), (size_t)f.gcount(), read_size);
        exit(EXIT_FAILURE);
    }
    if (max_bytes > 0 && (size_t)file_size > max_bytes) {
        printf("[data] %s is %.2f GB; capped read at %.2f GB (--max-bytes)\n",
               path.c_str(), file_size / 1e9, read_size / 1e9);
    }
    return buf;
}

inline std::string strip(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) a++;
    while (b > a && std::isspace((unsigned char)s[b - 1])) b--;
    return s.substr(a, b - a);
}

inline bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

inline Message to_message(const minijson::JsonValue& m) {
    Message msg;
    const minijson::JsonValue* r = m.get("role");
    const minijson::JsonValue* c = m.get("content");
    if (r && r->is_str()) msg.role = r->str;
    if (c && c->is_str()) msg.content = c->str;
    return msg;
}

inline Conversation messages_from_array(const minijson::JsonValue& arr) {
    Conversation conv;
    for (const auto& m : arr.arr) conv.push_back(to_message(m));
    return conv;
}

}  // namespace data_detail

// Load conversations from a .json or .jsonl file, matching the shapes accepted
// by utils/chat.py::load_conversations.
inline std::vector<Conversation> load_conversations(const std::string& path) {
    using minijson::JsonValue;
    std::string text = data_detail::strip(data_detail::read_file(path));

    std::vector<JsonValue> records;
    try {
        if (data_detail::ends_with(path, ".jsonl")) {
            std::istringstream lines(text);
            std::string line;
            while (std::getline(lines, line)) {
                if (data_detail::strip(line).empty()) continue;
                records.push_back(minijson::parse(line));
            }
        } else {
            JsonValue root = minijson::parse(text);
            if (root.is_obj()) records.push_back(std::move(root));
            else if (root.is_arr()) records = std::move(root.arr);
            else {
                fprintf(stderr, "[data] %s: top-level JSON must be an object or array\n", path.c_str());
                exit(EXIT_FAILURE);
            }
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "[data] failed to parse %s: %s\n", path.c_str(), e.what());
        exit(EXIT_FAILURE);
    }

    std::vector<Conversation> conversations;

    // A bare array of message dicts is a single conversation.
    if (!records.empty() && records[0].is_obj() && records[0].has("role")) {
        Conversation conv;
        for (const auto& m : records) conv.push_back(data_detail::to_message(m));
        conversations.push_back(std::move(conv));
        return conversations;
    }

    for (const auto& rec : records) {
        if (rec.is_obj()) {
            const JsonValue* msgs = rec.get("messages");
            if (!msgs || !msgs->is_arr()) {
                fprintf(stderr, "[data] %s: object without a 'messages' array\n", path.c_str());
                exit(EXIT_FAILURE);
            }
            conversations.push_back(data_detail::messages_from_array(*msgs));
        } else if (rec.is_arr()) {
            conversations.push_back(data_detail::messages_from_array(rec));
        } else {
            fprintf(stderr, "[data] %s: unexpected record type\n", path.c_str());
            exit(EXIT_FAILURE);
        }
    }
    return conversations;
}

namespace data_detail {

// Role tags, matching utils/chat.py::ROLE_TAGS.
inline std::string role_tag(const std::string& role) {
    if (role == "system") return "<|system|>";
    if (role == "user") return "<|user|>";
    if (role == "assistant") return "<|assistant|>";
    return "<|" + role + "|>";
}

inline void render_conversation(const DataTokenizer& tok, const Conversation& msgs, Dataset& d) {
    for (const Message& m : msgs) {
        std::vector<int> tag_ids = tok.encode(role_tag(m.role) + "\n");
        for (int id : tag_ids) {
            d.ids.push_back(id);
            d.mask.push_back(0);
        }
        std::vector<int> content_ids = tok.encode(m.content + "\n");
        char predict = (m.role == "assistant") ? 1 : 0;
        for (int id : content_ids) {
            d.ids.push_back(id);
            d.mask.push_back(predict);
        }
    }
}

}  // namespace data_detail

// Build a chat dataset: every conversation rendered with role tags and
// concatenated, only assistant-turn tokens marked as prediction targets.
inline Dataset load_chat_dataset(const DataTokenizer& tok, const std::string& path) {
    std::vector<Conversation> conversations = load_conversations(path);
    Dataset d;
    for (const Conversation& conv : conversations)
        data_detail::render_conversation(tok, conv, d);

    size_t n_targets = 0;
    for (char m : d.mask) n_targets += (m != 0);
    printf("Loaded %zu conversations, %zu/%zu tokens are assistant-turn prediction targets\n",
           conversations.size(), n_targets, d.mask.size());
    return d;
}

// Build a plain-text dataset: every token is a valid prediction target.
// max_bytes caps how much of the file is read (0 = unlimited).
//
// Streams the file in fixed-size chunks (split on the last newline in each
// chunk, so a run of BPE merges never spans a chunk boundary except for the
// whitespace immediately around that newline - negligible for a multi-GB
// corpus) instead of reading the whole file into one buffer and encoding it
// in one call. Reading-then-encoding the whole file at once needs the raw
// text buffer, the tokenizer's own output vector, and the vector<int> copy
// of it all resident simultaneously (~4x the corpus size), which is what
// OOM-killed a full run of a 10GB corpus on a 32GB box. Chunking bounds peak
// memory to one chunk's working set plus the final ids/mask arrays, grown in
// fixed increments (not exponential doubling) to avoid a reallocation spike
// near the end of a multi-billion-token run.
inline Dataset load_text_dataset(const DataTokenizer& tok, const std::string& path, size_t max_bytes = 0) {
    constexpr size_t CHUNK_BYTES = 256ull * 1024 * 1024;
    constexpr size_t GROW_STEP = 256ull * 1024 * 1024;  // elements, not bytes

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "[data] could not open %s\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    f.seekg(0, std::ios::end);
    std::streamoff file_size_off = f.tellg();
    if (file_size_off < 0) {
        fprintf(stderr, "[data] could not determine size of %s\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    size_t file_size = (size_t)file_size_off;
    size_t total_to_read = (max_bytes > 0 && max_bytes < file_size) ? max_bytes : file_size;
    if (max_bytes > 0 && file_size > max_bytes) {
        printf("[data] %s is %.2f GB; capped read at %.2f GB (--max-bytes)\n",
               path.c_str(), file_size / 1e9, total_to_read / 1e9);
    }
    f.seekg(0, std::ios::beg);

    auto ensure_capacity = [](auto& vec, size_t additional) {
        size_t needed = vec.size() + additional;
        if (needed > vec.capacity()) vec.reserve(std::max(needed, vec.capacity() + GROW_STEP));
    };

    Dataset d;
    std::string carry;  // bytes left over after the last newline of the previous chunk
    std::vector<char> buf(CHUNK_BYTES);
    size_t remaining = total_to_read;
    size_t chunk_idx = 0;

    while (remaining > 0) {
        size_t want = std::min(remaining, CHUNK_BYTES);
        f.read(buf.data(), (std::streamsize)want);
        size_t got = (size_t)f.gcount();
        if (got != want) {
            fprintf(stderr, "[data] short read on %s (%zu of %zu bytes in chunk %zu)\n",
                    path.c_str(), got, want, chunk_idx);
            exit(EXIT_FAILURE);
        }
        remaining -= got;

        std::string piece = carry + std::string(buf.data(), got);
        carry.clear();

        std::string chunk_text;
        if (remaining > 0) {
            // Hold back everything after the last newline so the next chunk's
            // encode() sees a whole word/whitespace-run, not half of one.
            size_t last_nl = piece.find_last_of('\n');
            if (last_nl == std::string::npos) {
                // No newline in this whole chunk (e.g. one huge line): carry
                // it all forward rather than splitting mid-token.
                carry = std::move(piece);
                chunk_idx++;
                continue;
            }
            carry = piece.substr(last_nl + 1);
            chunk_text = piece.substr(0, last_nl + 1);
        } else {
            chunk_text = std::move(piece);
        }

        std::vector<int> ids_chunk = tok.encode(chunk_text);
        ensure_capacity(d.ids, ids_chunk.size());
        d.ids.insert(d.ids.end(), ids_chunk.begin(), ids_chunk.end());
        chunk_idx++;
    }
    if (!carry.empty()) {
        std::vector<int> ids_chunk = tok.encode(carry);
        ensure_capacity(d.ids, ids_chunk.size());
        d.ids.insert(d.ids.end(), ids_chunk.begin(), ids_chunk.end());
    }

    d.mask.assign(d.ids.size(), 1);
    return d;
}
