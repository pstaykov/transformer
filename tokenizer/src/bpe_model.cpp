#include "bpe_model.hpp"
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <array>

namespace bbpe {

void BpeModel::init_byte_vocab() {
    id_to_token_.clear();
    id_to_token_.reserve(256);
    for (int b = 0; b < 256; ++b)
        id_to_token_.emplace_back(1, static_cast<char>(b));
    merge_pair_to_id_.clear();
    merges_.clear();
}

TokenId BpeModel::add_merge(TokenId a, TokenId b) {
    TokenId nid = static_cast<TokenId>(id_to_token_.size());
    id_to_token_.push_back(id_to_token_[a] + id_to_token_[b]);
    merge_pair_to_id_[pair_key(a, b)] = nid;
    merges_.emplace_back(a, b);
    return nid;
}

void BpeModel::rebuild_from_merges(
    const std::vector<std::pair<TokenId,TokenId>>& merges) {
    init_byte_vocab();
    for (const auto& [a,b] : merges) add_merge(a, b);
}

std::vector<TokenId> BpeModel::encode_word(const std::string& wb) const {
    if (wb.empty()) return {};
    std::vector<TokenId> ids;
    ids.reserve(wb.size());
    for (unsigned char c : wb) ids.push_back(static_cast<TokenId>(c));

    while (ids.size() > 1) {
        TokenId best_id = -1;
        size_t  best_pos = static_cast<size_t>(-1);
        for (size_t i = 0; i + 1 < ids.size(); ++i) {
            auto it = merge_pair_to_id_.find(pair_key(ids[i], ids[i+1]));
            if (it != merge_pair_to_id_.end() &&
                (best_pos == static_cast<size_t>(-1) || it->second < best_id)) {
                best_id  = it->second;
                best_pos = i;
            }
        }
        if (best_pos == static_cast<size_t>(-1)) break;
        ids[best_pos] = best_id;
        ids.erase(ids.begin() + static_cast<long>(best_pos) + 1);
    }
    return ids;
}

void Tokenizer::add_special_token(const std::string& text, TokenId id) {
    special_text_to_id_[text] = id;
    special_id_to_text_[id]   = text;
    specials_by_len_desc_.push_back(text);
    std::sort(specials_by_len_desc_.begin(), specials_by_len_desc_.end(),
              [](const std::string& a, const std::string& b){ return a.size() > b.size(); });
}

std::vector<TokenId> Tokenizer::encode(const std::string& text) const {
    std::vector<TokenId> result;
    result.reserve(text.size() / 3 + 4);

    auto encode_segment = [&](const std::string& seg) {
        for (const std::string& piece : Pretokenizer::split(seg))
            for (TokenId id : model_.encode_word(piece))
                result.push_back(id);
    };

    size_t i = 0, normal_start = 0;
    while (i < text.size()) {
        const std::string* match = nullptr;
        for (const auto& sp : specials_by_len_desc_) {
            if (i + sp.size() <= text.size() &&
                text.compare(i, sp.size(), sp) == 0) {
                match = &sp; break;
            }
        }
        if (match) {
            if (i > normal_start)
                encode_segment(text.substr(normal_start, i - normal_start));
            result.push_back(special_text_to_id_.at(*match));
            i += match->size();
            normal_start = i;
        } else {
            ++i;
        }
    }
    if (normal_start < text.size())
        encode_segment(text.substr(normal_start));

    return result;
}

std::string Tokenizer::decode(const std::vector<TokenId>& ids,
                               bool skip_special_tokens) const {
    std::string out;
    out.reserve(ids.size() * 3);
    for (TokenId id : ids) {
        if (auto it = special_id_to_text_.find(id); it != special_id_to_text_.end()) {
            if (!skip_special_tokens) out += it->second;
        } else if (id >= 0 && static_cast<size_t>(id) < model_.vocab_size()) {
            out += model_.token_bytes(id);
        }
    }
    return out;
}

namespace {
void w32(std::ofstream& f, uint32_t v) { f.write(reinterpret_cast<const char*>(&v),4); }
uint32_t r32(std::ifstream& f) {
    uint32_t v=0; f.read(reinterpret_cast<char*>(&v),4); return v;
}
} // namespace

void Tokenizer::save_binary(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Schreiben fehlgeschlagen: " + path);

    f.write("BBPEV1\0\0", 8);
    const auto& M = model_.merges();
    w32(f, static_cast<uint32_t>(M.size()));
    for (const auto& [a,b] : M) { w32(f,static_cast<uint32_t>(a)); w32(f,static_cast<uint32_t>(b)); }

    std::vector<TokenId> sids;
    sids.reserve(special_id_to_text_.size());
    for (const auto& [id,txt] : special_id_to_text_) { (void)txt; sids.push_back(id); }
    std::sort(sids.begin(), sids.end());

    w32(f, static_cast<uint32_t>(sids.size()));
    for (TokenId id : sids) {
        const std::string& txt = special_id_to_text_.at(id);
        w32(f, static_cast<uint32_t>(id));
        w32(f, static_cast<uint32_t>(txt.size()));
        f.write(txt.data(), static_cast<std::streamsize>(txt.size()));
    }
}

Tokenizer Tokenizer::load_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Öffnen fehlgeschlagen: " + path);

    char magic[8]; f.read(magic,8);
    if (std::string(magic,6) != "BBPEV1")
        throw std::runtime_error("Ungültiges Dateiformat: " + path);

    Tokenizer tok;
    uint32_t nm = r32(f);
    std::vector<std::pair<TokenId,TokenId>> merges(nm);
    for (auto& [a,b] : merges) {
        a = static_cast<TokenId>(r32(f));
        b = static_cast<TokenId>(r32(f));
    }
    tok.model_.rebuild_from_merges(merges);

    uint32_t ns = r32(f);
    for (uint32_t i = 0; i < ns; ++i) {
        TokenId id = static_cast<TokenId>(r32(f));
        uint32_t len = r32(f);
        std::string txt(len, '\0');
        f.read(txt.data(), len);
        tok.add_special_token(txt, id);
    }
    return tok;
}

namespace {
std::array<uint32_t,256> byte_to_unicode() {
    std::array<uint32_t,256> t{};
    std::vector<int> pr;
    for (int b=33;b<=126;++b) pr.push_back(b);
    for (int b=161;b<=172;++b) pr.push_back(b);
    for (int b=174;b<=255;++b) pr.push_back(b);
    for (int b : pr) t[b]=static_cast<uint32_t>(b);
    int next=256;
    for (int b=0;b<256;++b) {
        if (std::find(pr.begin(),pr.end(),b)==pr.end()) t[b]=static_cast<uint32_t>(next++);
    }
    return t;
}
std::string cp_to_utf8(uint32_t cp) {
    std::string o;
    if (cp<0x80) { o+=static_cast<char>(cp); }
    else if (cp<0x800) { o+=static_cast<char>(0xC0|(cp>>6)); o+=static_cast<char>(0x80|(cp&0x3F)); }
    else { o+=static_cast<char>(0xE0|(cp>>12)); o+=static_cast<char>(0x80|((cp>>6)&0x3F)); o+=static_cast<char>(0x80|(cp&0x3F)); }
    return o;
}
std::string to_printable(const std::string& raw) {
    static const auto T=byte_to_unicode();
    std::string o; for (unsigned char b:raw) o+=cp_to_utf8(T[b]); return o;
}
std::string jesc(const std::string& s) {
    std::string o;
    for (char c:s) { if(c=='"') o+="\\\""; else if(c=='\\') o+="\\\\"; else o+=c; }
    return o;
}
} // namespace

void Tokenizer::export_human_readable(const std::string& dir) const {
    {
        std::ofstream f(dir+"/vocab.json");
        f << "{\n";
        const auto& T=model_.id_to_token();
        for (size_t id=0;id<T.size();++id) {
            f << "  \"" << jesc(to_printable(T[id])) << "\": " << id;
            f << (id+1<T.size() ? ",\n" : "\n");
        }
        f << "}\n";
    }
    {
        std::ofstream f(dir+"/merges.txt");
        f << "#version: 0.2\n";
        const auto& T=model_.id_to_token();
        for (const auto& [a,b]:model_.merges())
            f << to_printable(T[a]) << " " << to_printable(T[b]) << "\n";
    }
}

} // namespace bbpe
