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

    // Re-point an existing special-token id at new text, e.g. to repurpose a
    // reserved slot for a different tag. Unlike add_special_token(), this also
    // drops the id's old text from the encode-side lookup (specials_by_len_desc_
    // and special_text_to_id_), so the old text stops being recognized as a
    // special once renamed rather than lingering as a stale alias for the id.
    void remap_special_token(const std::string& new_text, TokenId id);

    std::vector<TokenId> encode(const std::string& text) const;
    std::string decode(const std::vector<TokenId>& ids,
                       bool skip_special_tokens = true) const;

    size_t vocab_size_regular() const { return model_.vocab_size(); }
    size_t vocab_size_total()   const {
        return model_.vocab_size() + special_id_to_text_.size();
    }

    const std::unordered_map<TokenId, std::string>& special_tokens() const {
        return special_id_to_text_;
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
