#include "bpe_trainer.hpp"
#include "pretokenizer.hpp"
#include "common.hpp"

#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <iostream>
#include <algorithm>
#include <chrono>

namespace bbpe {

std::vector<std::pair<std::string, uint64_t>>
BpeTrainer::count_words(const CorpusReader& reader) const {

    std::unordered_map<std::string, uint64_t> counts;
    counts.reserve(1 << 20);

    uint64_t total = reader.estimate_total_bytes();
    uint64_t done  = 0;
    uint64_t docs  = 0;

    // Sofortige Rückmeldung, dass das Programm gestartet ist und nicht hängt.
    std::cerr << "Starte Phase 1: Streaming + Wortzählung ...\n";
    ProgressBar bar(total, "Zähle Woerter (streaming)");
    bar.update(0);

    reader.for_each_document([&](const std::string& text) {
        for (const std::string& piece : Pretokenizer::split(text))
            ++counts[piece];
        done += text.size() + 1;
        ++docs;
        // Häufigeres Update (alle 10.000 statt 50.000 Dokumente) für
        // schnelleres visuelles Feedback, insbesondere bei sehr vielen
        // kurzen Zeilen (z.B. 50 Mio. Zeilen bei 10 GB Korpus).
        if (docs % 10'000 == 0) bar.update(done);
    });
    bar.finish();

    std::cerr << "Dokumente verarbeitet  : " << docs << "\n";
    std::cerr << "Eindeutige Pretokens   : " << counts.size() << "\n";

    std::vector<std::pair<std::string, uint64_t>> result;
    result.reserve(counts.size());
    for (auto& [w, c] : counts) {
        if (c >= config_.min_frequency)
            result.emplace_back(std::move(w), c);
    }
    counts.clear();
    std::cerr << "Nach min_freq-Filter   : " << result.size() << " Pretokens\n";
    return result;
}

namespace {
struct HeapItem {
    int64_t count;
    TokenId a, b;
    bool operator<(const HeapItem& o) const {
        if (count != o.count) return count < o.count;
        if (a != o.a) return a > o.a;
        return b > o.b;
    }
};
struct WordEntry {
    std::vector<TokenId> syms;
    uint64_t count;
};
} // namespace

BpeModel BpeTrainer::train(const CorpusReader& reader) const {
    BpeModel model;
    model.init_byte_vocab();

    auto word_counts = count_words(reader);

    std::cerr << "Starte Phase 2: BPE-Merge-Loop ...\n";

    std::vector<WordEntry> words;
    words.reserve(word_counts.size());
    for (auto& [ws, cnt] : word_counts) {
        WordEntry e;
        e.syms.reserve(ws.size());
        for (unsigned char c : ws) e.syms.push_back(static_cast<TokenId>(c));
        e.count = cnt;
        words.push_back(std::move(e));
    }
    word_counts.clear();
    word_counts.shrink_to_fit();

    std::unordered_map<uint64_t, int64_t>                       pair_counts;
    std::unordered_map<uint64_t, std::unordered_set<uint32_t>>  pair_to_words;
    pair_counts.reserve(1 << 20);
    pair_to_words.reserve(1 << 20);

    for (uint32_t idx = 0; idx < words.size(); ++idx) {
        const auto& s = words[idx].syms;
        const int64_t c = static_cast<int64_t>(words[idx].count);
        for (size_t i = 0; i + 1 < s.size(); ++i) {
            uint64_t k = pair_key(s[i], s[i+1]);
            pair_counts[k] += c;
            pair_to_words[k].insert(idx);
        }
    }

    std::priority_queue<HeapItem> heap;
    for (const auto& [k, c] : pair_counts)
        heap.push({c, pair_first(k), pair_second(k)});

    const size_t target = config_.vocab_size_regular > 256
                        ? config_.vocab_size_regular - 256 : 0;
    std::unordered_set<uint64_t> banned;

    ProgressBar bar(target, "BPE-Merges");
    size_t done = 0;
    auto t0 = std::chrono::steady_clock::now();

    while (done < target && !heap.empty()) {
        HeapItem top = heap.top(); heap.pop();
        uint64_t key = pair_key(top.a, top.b);

        auto it = pair_counts.find(key);
        if (it == pair_counts.end() || it->second != top.count) continue;
        if (banned.count(key)) continue;
        if (top.count < static_cast<int64_t>(config_.min_frequency)) break;

        if (model.token_bytes(top.a).size() +
            model.token_bytes(top.b).size() > config_.max_token_length) {
            banned.insert(key);
            pair_counts.erase(key);
            continue;
        }

        TokenId new_id = model.add_merge(top.a, top.b);
        ++done;

        auto wit = pair_to_words.find(key);
        if (wit == pair_to_words.end()) continue;
        std::vector<uint32_t> affected(wit->second.begin(), wit->second.end());
        pair_to_words.erase(wit);
        pair_counts.erase(key);

        for (uint32_t widx : affected) {
            WordEntry& w = words[widx];
            const int64_t wc = static_cast<int64_t>(w.count);

            for (size_t i = 0; i + 1 < w.syms.size(); ++i) {
                uint64_t k = pair_key(w.syms[i], w.syms[i+1]);
                auto pi = pair_counts.find(k);
                if (pi != pair_counts.end()) {
                    pi->second -= wc;
                    if (pi->second <= 0) pair_counts.erase(pi);
                }
            }

            std::vector<TokenId> merged;
            merged.reserve(w.syms.size());
            for (size_t i = 0; i < w.syms.size(); ) {
                if (i+1 < w.syms.size() &&
                    w.syms[i] == top.a && w.syms[i+1] == top.b) {
                    merged.push_back(new_id); i += 2;
                } else {
                    merged.push_back(w.syms[i]); ++i;
                }
            }
            w.syms = std::move(merged);

            for (size_t i = 0; i + 1 < w.syms.size(); ++i) {
                uint64_t k = pair_key(w.syms[i], w.syms[i+1]);
                int64_t nc = (pair_counts[k] += wc);
                pair_to_words[k].insert(widx);
                heap.push({nc, w.syms[i], w.syms[i+1]});
            }
        }

        if (done % 200 == 0 || done == target) bar.update(done);
    }
    bar.finish();

    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::cerr << "Merges abgeschlossen   : " << done
              << " in " << elapsed/60.0 << " min\n";

    if (done < target)
        std::cerr << "[WARN] Ziel nicht erreicht ("
                  << (256+done) << "/" << config_.vocab_size_regular
                  << ") — Korpus zu klein oder min_frequency zu hoch.\n";

    return model;
}

} // namespace bbpe
