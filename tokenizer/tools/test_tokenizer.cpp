#include "bpe_model.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

using namespace bbpe;

struct Result { std::string label; size_t bytes, tokens; };

Result measure(const Tokenizer& tok, const std::string& label, const std::string& text) {
    return {label, text.size(), tok.encode(text).size()};
}

int main(int argc, char** argv) {
    std::string path = "./tok_out/tokenizer.bbpe";
    for (int i=1;i<argc;++i) {
        if (std::string(argv[i])=="--tokenizer-path"&&i+1<argc) path=argv[++i];
    }

    std::cerr << "Lade: " << path << "\n\n";
    Tokenizer tok = Tokenizer::load_binary(path);

    std::cout << "Regulaere Tokens : " << tok.vocab_size_regular() << "\n";
    std::cout << "Gesamt-Tokens    : " << tok.vocab_size_total()   << "\n";
    std::cout << std::string(65,'=') << "\n";

    std::string demo = "Hallo Welt! Hello World! ni hao! <|endoftext|>";
    auto ids = tok.encode(demo);
    auto dec = tok.decode(ids);
    std::cout << "\nDemo      : " << demo
              << "\nDecoded   : " << dec
              << "\nRoundtrip : " << (dec == demo ? "OK" : "FEHLER") << "\n";

    static const std::string DE =
        "Die kuenstliche Intelligenz veraendert grundlegend, wie wir arbeiten, "
        "kommunizieren und komplexe Probleme loesen.";
    static const std::string EN =
        "Artificial intelligence is fundamentally transforming how we work, "
        "communicate, and solve complex problems with high precision.";
    static const std::string CODE =
        "def fibonacci(n: int) -> int:\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n): a, b = b, a+b\n"
        "    return a\n";

    std::vector<Result> R = {
        measure(tok,"Deutsch",DE),
        measure(tok,"Englisch",EN),
        measure(tok,"Code",CODE),
    };

    std::cout << "\n" << std::string(65,'=') << "\n";
    std::cout << "KOMPRESSION (hoeher = effizienter)\n";
    std::cout << std::string(65,'-') << "\n";
    std::cout << std::left  << std::setw(14) << "Sprache"
              << std::right << std::setw(8)  << "Bytes"
              << std::setw(8)  << "Tokens"
              << std::setw(13) << "Bytes/Token" << "\n";
    std::cout << std::string(43,'-') << "\n";
    for (auto& r : R) {
        double bpt = r.tokens ? static_cast<double>(r.bytes)/r.tokens : 0;
        std::cout << std::left  << std::setw(14) << r.label
                  << std::right << std::setw(8)  << r.bytes
                  << std::setw(8)  << r.tokens
                  << std::setw(13) << std::fixed << std::setprecision(3) << bpt << "\n";
    }

    std::string rob = "Test mit Sonderzeichen und mixed scripts: kyrillica cirillico";
    auto rob_ids = tok.encode(rob);
    auto rob_dec = tok.decode(rob_ids);
    std::cout << "\n" << std::string(65,'=') << "\n";
    std::cout << "ROBUSTHEITSTEST\n";
    std::cout << "Input     : " << rob << "\n";
    std::cout << "Tokens    : " << rob_ids.size() << "\n";
    std::cout << "Roundtrip : " << (rob_dec==rob ? "OK (kein OOV)" : "FEHLER") << "\n";
    return 0;
}
