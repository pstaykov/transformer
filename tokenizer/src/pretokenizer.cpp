#include "pretokenizer.hpp"
#include "unicode_utils.hpp"
#include <vector>
#include <string>

namespace bbpe {

using unicode::decode_utf8;
using unicode::is_letter;
using unicode::is_number;
using unicode::is_whitespace;

std::vector<std::string> Pretokenizer::split(const std::string& text) {
    std::vector<std::string> pieces;
    const size_t n = text.size();
    size_t i = 0;

    while (i < n) {
        auto [cp, len] = decode_utf8(text, i);
        if (len == 0) {
            i++; // Schutz vor invalidem UTF-8
            continue;
        }

        // 1. KONTRAKTIONEN (GPT-4 Style)
        // Erfasst optionales Apostroph mit Buchstaben (z.B. 's, 't, oder im Spanischen/Französischen d', l')
        if (text[i] == '\'' && i + 1 < n) {
            auto [next_cp, next_len] = decode_utf8(text, i + 1);
            if (is_letter(next_cp)) {
                size_t j = i + 1 + next_len;
                // Maximal noch einen Buchstaben mitnehmen (z.B. 'll, 're)
                if (j < n) {
                    auto [after_cp, after_len] = decode_utf8(text, j);
                    if (is_letter(after_cp)) j += after_len;
                }
                pieces.emplace_back(text.substr(i, j - i));
                i = j;
                continue;
            }
        }

        // 2. BUCHSTABEN (Beliebige Sprache / UTF-8 kompatibel)
        if (is_letter(cp)) {
            size_t j = i + len;
            while (j < n) {
                auto [c, l] = decode_utf8(text, j);
                if (!is_letter(c)) break;
                j += l;
            }
            pieces.emplace_back(text.substr(i, j - i));
            i = j;
            continue;
        }

        // 3. ZAHLEN (GPT-4 Limitierung: Maximal 3 Ziffern am Stück)
        // Das ist der Gamechanger für Mathe und Code! Macht aus 123456 -> 123 und 456.
        if (is_number(cp)) {
            size_t j = i + len;
            size_t digit_count = 1;
            while (j < n && digit_count < 3) {
                auto [c, l] = decode_utf8(text, j);
                if (!is_number(c)) break;
                j += l;
                digit_count++;
            }
            pieces.emplace_back(text.substr(i, j - i));
            i = j;
            continue;
        }

        // 4. LEERZEICHEN & CODE-EINRÜCKUNGEN (Spaces / Tabs)
        // GPT-4 fasst Leerzeichen zusammen, trennt sie aber vor dem nächsten Nicht-Leerzeichen auf.
        // Wichtig für Code: Tabulatoren oder Spaces werden präzise gruppiert.
        if (is_whitespace(cp)) {
            size_t j = i + len;
            while (j < n) {
                auto [c, l] = decode_utf8(text, j);
                if (!is_whitespace(c)) break;
                j += l;
            }
            
            // Wenn danach noch Text kommt, lassen wir das letzte Leerzeichen für das nächste Wort übrig
            // (Das ist das klassische BPE-Verschmelzungs-Feature)
            if (j < n) {
                // Wir müssen ein Zeichen zurückgehen (UTF-8 sicher)
                size_t prev = i;
                size_t curr = i;
                while (curr < j) {
                    auto [_, l] = decode_utf8(text, curr);
                    prev = curr;
                    curr += l;
                }
                if (prev > i) j = prev; 
            }
            
            pieces.emplace_back(text.substr(i, j - i));
            i = j;
            continue;
        }

        // 5. SONDERZEICHEN / CODE-OPERATOREN
        // Für Code extrem wichtig: Operatoren wie `==`, `++`, `!=` oder `->` sollten 
        // nicht in riesigen Clustern landen, sondern einzeln oder in logischen Paaren stehen.
        // GPT-4 trennt Satzzeichen/Symbole meist strikt einzeln ab.
        pieces.emplace_back(text.substr(i, len));
        i += len;
    }

    return pieces;
}

} // namespace bbpe