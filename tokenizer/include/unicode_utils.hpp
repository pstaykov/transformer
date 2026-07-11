#pragma once
#include <cstdint>
#include <string>
#include <utility>

namespace bbpe::unicode {

std::pair<uint32_t, size_t> decode_utf8(const std::string& text, size_t pos) noexcept;

bool is_letter(uint32_t cp) noexcept;
bool is_number(uint32_t cp) noexcept;
bool is_whitespace(uint32_t cp) noexcept;

} // namespace bbpe::unicode
