#include "bpe_trainer.hpp"
#include "corpus_reader.hpp"
#include "bpe_model.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

using namespace bbpe;

static const std::vector<std::string> SPECIAL_TOKENS = {
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
};

struct Args {
    std::vector<std::string> data_dirs;
    std::string output_dir;
    std::string text_field = "text";
    size_t vocab_size = 151'643;
    uint64_t min_frequency = 20;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto nxt = [&]{ return (i+1<argc) ? std::string(argv[++i]) : std::string(); };
        if      (s=="--data-dir")    { a.data_dirs.push_back(nxt());
                                       while(i+1<argc&&argv[i+1][0]!='-') a.data_dirs.push_back(nxt()); }
        else if (s=="--output-dir")  a.output_dir  = nxt();
        else if (s=="--text-field")  a.text_field  = nxt();
        else if (s=="--vocab-size")  a.vocab_size  = std::stoul(nxt());
        else if (s=="--min-frequency") a.min_frequency = std::stoull(nxt());
        else std::cerr << "[WARN] Unbekanntes Argument: " << s << "\n";
    }
    if (a.data_dirs.empty()||a.output_dir.empty())
        throw std::runtime_error("--data-dir und --output-dir erforderlich.");
    return a;
}

int main(int argc, char** argv) {
    try {
        auto args = parse_args(argc, argv);
        std::filesystem::create_directories(args.output_dir);

        CorpusReader reader(args.data_dirs, args.text_field);
        uint64_t total_mb = reader.estimate_total_bytes() / (1024*1024);
        std::cerr << "Korpusgroesse (geschaetzt): " << total_mb << " MB\n";

        TrainerConfig cfg;
        cfg.vocab_size_regular = args.vocab_size;
        cfg.text_field         = args.text_field;
        cfg.min_frequency      = args.min_frequency;

        BpeTrainer trainer(cfg);
        BpeModel model = trainer.train(reader);

        Tokenizer tok;
        tok.model() = std::move(model);

        TokenId next_id = static_cast<TokenId>(tok.vocab_size_regular());
        for (const auto& s : SPECIAL_TOKENS)
            tok.add_special_token(s, next_id++);

        std::cerr << "Regulaere Tokens : " << tok.vocab_size_regular() << "\n";
        std::cerr << "Gesamt (inkl. Special) : " << tok.vocab_size_total() << "\n";

        std::string bin = args.output_dir + "/tokenizer.bbpe";
        tok.save_binary(bin);
        std::cerr << "Gespeichert     : " << bin << "\n";

        tok.export_human_readable(args.output_dir);
        std::cerr << "Exportiert      : vocab.json, merges.txt\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[FEHLER] " << e.what() << "\n";
        return 1;
    }
}
