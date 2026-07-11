#include "unicode_utils.hpp"

namespace bbpe::unicode {

std::pair<uint32_t, size_t> decode_utf8(const std::string& t, size_t pos) noexcept {
    if (pos >= t.size()) return {0, 0};
    auto u = [&](size_t off) -> int {
        if (pos + off >= t.size()) return -1;
        unsigned char c = static_cast<unsigned char>(t[pos + off]);
        return (c & 0xC0) == 0x80 ? (c & 0x3F) : -1;
    };
    unsigned char c0 = static_cast<unsigned char>(t[pos]);
    if (c0 < 0x80) return {c0, 1};
    if ((c0 & 0xE0) == 0xC0) {
        int c1 = u(1);
        if (c1 < 0) return {c0, 1};
        return {((c0 & 0x1Fu) << 6) | static_cast<uint32_t>(c1), 2};
    }
    if ((c0 & 0xF0) == 0xE0) {
        int c1 = u(1), c2 = u(2);
        if (c1 < 0 || c2 < 0) return {c0, 1};
        return {((c0 & 0x0Fu) << 12) | (static_cast<uint32_t>(c1) << 6) |
                static_cast<uint32_t>(c2), 3};
    }
    if ((c0 & 0xF8) == 0xF0) {
        int c1 = u(1), c2 = u(2), c3 = u(3);
        if (c1 < 0 || c2 < 0 || c3 < 0) return {c0, 1};
        return {((c0 & 0x07u) << 18) | (static_cast<uint32_t>(c1) << 12) |
                (static_cast<uint32_t>(c2) << 6) | static_cast<uint32_t>(c3), 4};
    }
    return {c0, 1};
}

bool is_whitespace(uint32_t cp) noexcept {
    switch (cp) {
        case ' ': case '\t': case '\n': case '\r':
        case '\v': case '\f': case 0x00A0: case 0x1680:
        case 0x2028: case 0x2029: case 0x202F:
        case 0x205F: case 0x3000: return true;
    }
    return cp >= 0x2000 && cp <= 0x200A;
}

bool is_number(uint32_t cp) noexcept {
    return (cp >= '0' && cp <= '9') || (cp >= 0xFF10 && cp <= 0xFF19);
}

bool is_letter(uint32_t cp) noexcept {
    if ((cp>='a'&&cp<='z')||(cp>='A'&&cp<='Z')) return true;
    if (cp >= 0x00AA && cp <= 0x02AF) return true;
    if (cp >= 0x0370 && cp <= 0x04FF) return true;
    if (cp >= 0x0590 && cp <= 0x08FF) return true;
    if (cp >= 0x0900 && cp <= 0x0DFF) return true;
    if (cp >= 0x1E00 && cp <= 0x1EFF) return true;
    if (cp >= 0x3040 && cp <= 0x30FF) return true;
    if (cp >= 0x3400 && cp <= 0x9FFF) return true;
    if (cp >= 0xAC00 && cp <= 0xD7A3) return true;
    if (cp >= 0xF900 && cp <= 0xFAFF) return true;
    return false;
}

} // namespace bbpe::unicode
