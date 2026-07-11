#pragma once
#include <string>
#include <vector>
#include "bpe_model.hpp"
#include "corpus_reader.hpp"

namespace bbpe {

struct TrainerConfig {
    size_t   vocab_size_regular = 151'643;
    uint64_t min_frequency      = 20;
    size_t   max_token_length   = 16;
    std::string text_field      = "text";
};

class BpeTrainer {
public:
    explicit BpeTrainer(TrainerConfig config) : config_(std::move(config)) {}

    BpeModel train(const CorpusReader& reader) const;

private:
    TrainerConfig config_;

    std::vector<std::pair<std::string, uint64_t>>
    count_words(const CorpusReader& reader) const;
};

} // namespace bbpe
