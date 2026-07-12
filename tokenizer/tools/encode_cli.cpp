/**
 * encode_cli.cpp
 *
 * Einfaches CLI zum Testen von encode()/decode() mit beliebigem Text.
 * Kann auch als Python-Modul kompiliert werden.
 *
 * CLI-Nutzung:
 *   ./encode_cli --tokenizer-path ./tok_out/tokenizer.bbpe --text "Hello world"
 */
#include "bpe_model.hpp"
#include <iostream>
#include <string>

using namespace bbpe;

// Die main-Funktion wird nur kompiliert, wenn wir das C++ CLI bauen.
// Für das Python-Modul wird sie ignoriert.
#ifndef PYBIND_BUILD
int main(int argc, char** argv) {
    std::string path = "./tok_out/tokenizer.bbpe";
    std::string text;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--tokenizer-path" && i+1 < argc) path = argv[++i];
        else if (a == "--text" && i+1 < argc) text = argv[++i];
    }

    if (text.empty()) {
        std::cerr << "Nutzung: " << argv[0]
                  << " --tokenizer-path <pfad> --text \"dein Satz\"\n";
        return 1;
    }

    Tokenizer tok = Tokenizer::load_binary(path);

    auto ids = tok.encode(text);
    auto decoded = tok.decode(ids);

    std::cout << "Text        : " << text << "\n";
    std::cout << "Anzahl Bytes: " << text.size() << "\n";
    std::cout << "Anzahl Tokens: " << ids.size() << "\n";
    std::cout << "Bytes/Token : "
              << (ids.empty() ? 0.0 : static_cast<double>(text.size()) / ids.size())
              << "\n\n";

    std::cout << "Token-IDs   : ";
    for (auto id : ids) std::cout << id << " ";
    std::cout << "\n\n";

    std::cout << "Einzelne Tokens (als Bytes/Text):\n";
    for (auto id : ids) {
        if (static_cast<size_t>(id) < tok.model().vocab_size()) {
            std::cout << "  [" << id << "] \""
                      << tok.model().token_bytes(id) << "\"\n";
        } else {
            std::cout << "  [" << id << "] (Sondertoken)\n";
        }
    }

    std::cout << "\nDecoded     : " << decoded << "\n";
    std::cout << "Roundtrip OK: " << (decoded == text ? "ja" : "NEIN") << "\n";

    return 0;
}
#endif // PYBIND_BUILD

// ============================================================================
// Python-Bindings (pybind11)
// ============================================================================
#ifdef PYBIND_BUILD
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Ermöglicht automatische Konvertierung von std::vector in Python-Listen

namespace py = pybind11;
PYBIND11_MODULE(bbpe_tokenizer, m) {
    m.doc() = "Python-Bindings für den C++ Byte-Pair-Encoding Tokenizer";

    py::class_<Tokenizer>(m, "Tokenizer")
        .def_static("load_binary", &Tokenizer::load_binary,
                    py::arg("path"), "Lädt ein binäres Tokenizer-Modell")

        // Falls encode in deiner bpe_model.hpp auch zwei Parameter hat,
        // musst du es hier ebenfalls anpassen. Ich gehe erstmal von einem aus:
        .def("encode", &Tokenizer::encode,
                    py::arg("text"), "Codiert einen Text in Token-IDs")

        // HIER IST DIE ÄNDERUNG: Wir fügen das zweite py::arg für den bool hinzu!
        // Du kannst den Namen "skip_special" anpassen, je nachdem was der bool macht.
        .def("decode", &Tokenizer::decode,
                    py::arg("ids"), py::arg("skip_special") = true,
                    "Decodiert Token-IDs zurück in Text");
}
#endif // PYBIND_BUILD