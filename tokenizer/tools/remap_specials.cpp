// One-off migration: repoint a tokenizer's reserved special-token ids at new
// text without touching its BPE merges or regular vocab.
//
// tok_out_kevindata/tokenizer.bbpe reserves 5 special ids (32000-32004) that
// were never actually used by the training pipeline - they're leftover
// ChatML-style tags (<|endoftext|>, <|im_start|>, <|im_end|>, <tool_call>,
// </tool_call>) from an earlier format, while utils/chat.py's real role tags
// (<|system|>, <|user|>, <|assistant|>) have always been encoded as ordinary
// multi-token BPE text. This repurposes 3 of those 5 already-reserved slots
// for the real role tags so a future SFT run can have them tokenize as single
// special tokens instead - no vocab_size change, so it stays compatible with
// the checkpoint's embedding table size, but note the model has never seen
// these ids used this way, so existing checkpoints need retraining before
// this has any effect.
//
// Usage: remap_specials <tokenizer.bbpe> <output.bbpe> <id0>=<text0> [<id1>=<text1> ...]
#include "bpe_model.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <in.bbpe> <out.bbpe> <id>=<text> [<id>=<text> ...]\n", argv[0]);
        return 1;
    }
    std::string in_path = argv[1];
    std::string out_path = argv[2];

    bbpe::Tokenizer tok = bbpe::Tokenizer::load_binary(in_path);

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        size_t eq = arg.find('=');
        if (eq == std::string::npos) {
            fprintf(stderr, "bad argument (expected id=text): %s\n", arg.c_str());
            return 1;
        }
        bbpe::TokenId id = (bbpe::TokenId)std::stoi(arg.substr(0, eq));
        std::string text = arg.substr(eq + 1);
        tok.remap_special_token(text, id);
        fprintf(stderr, "remapped id %d -> \"%s\"\n", id, text.c_str());
    }

    tok.save_binary(out_path);

    fprintf(stderr, "wrote %s (regular=%zu total=%zu)\n",
            out_path.c_str(), tok.vocab_size_regular(), tok.vocab_size_total());
    return 0;
}
