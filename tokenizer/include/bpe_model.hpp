#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include "common.hpp"
#include "pretokenizer.hpp"

namespace bbpe {

class BpeModel {
public:
    void init_byte_vocab();
    TokenId add_merge(TokenId a, TokenId b);
    std::vector<TokenId> encode_word(const std::string& word_bytes) const;
    void rebuild_from_merges(
        const std::vector<std::pair<TokenId, TokenId>>& merges);

    const std::string& token_bytes(TokenId id) const {
        return id_to_token_[static_cast<size_t>(id)];
    }
    size_t vocab_size() const { return id_to_token_.size(); }
    const std::vector<std::pair<TokenId,TokenId>>& merges() const { return merges_; }
    const std::vector<std::string>& id_to_token() const { return id_to_token_; }

private:
    std::vector<std::string>                  id_to_token_;
    std::unordered_map<uint64_t, TokenId>     merge_pair_to_id_;
    std::vector<std::pair<TokenId, TokenId>>  merges_;
};

class Tokenizer {
public:
    void add_special_token(const std::string& text, TokenId id);

    std::vector<TokenId> encode(const std::string& text) const;
    std::string decode(const std::vector<TokenId>& ids,
                       bool skip_special_tokens = true) const;

    size_t vocab_size_regular() const { return model_.vocab_size(); }
    size_t vocab_size_total()   const {
        return model_.vocab_size() + special_id_to_text_.size();
    }

    BpeModel&       model()       { return model_; }
    const BpeModel& model() const { return model_; }

    void save_binary(const std::string& path) const;
    static Tokenizer load_binary(const std::string& path);
    void export_human_readable(const std::string& dir) const;

private:
    BpeModel model_;
    std::unordered_map<std::string, TokenId> special_text_to_id_;
    std::unordered_map<TokenId, std::string> special_id_to_text_;
    std::vector<std::string> specials_by_len_desc_;
};

} // namespace bbpe
