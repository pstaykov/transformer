#pragma once
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <string>
#include <algorithm>

namespace bbpe {

using TokenId = int32_t;

inline uint64_t pair_key(TokenId a, TokenId b) noexcept {
    return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
            static_cast<uint64_t>(static_cast<uint32_t>(b));
}

inline TokenId pair_first(uint64_t key) noexcept {
    return static_cast<TokenId>(key >> 32);
}

inline TokenId pair_second(uint64_t key) noexcept {
    return static_cast<TokenId>(key & 0xFFFFFFFFu);
}

class ProgressBar {
public:
    explicit ProgressBar(uint64_t total, std::string label = "", int width = 45)
        : total_(total == 0 ? 1 : total),
          label_(std::move(label)),
          width_(width),
          start_(std::chrono::steady_clock::now()) {}

    void update(uint64_t current) {
        current = std::min(current, total_);
        double frac = static_cast<double>(current) / static_cast<double>(total_);
        int filled = static_cast<int>(frac * width_);

        auto now  = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_).count();
        double rate     = elapsed > 0.0 ? current / elapsed : 0.0;

        int elapsed_s = static_cast<int>(elapsed);
        int mm = elapsed_s / 60;
        int ss = elapsed_s % 60;

        std::fprintf(stderr, "\r%-28s [", label_.c_str());
        for (int i = 0; i < width_; ++i)
            std::fputc(i < filled ? '#' : '-', stderr);
        std::fprintf(stderr, "] %5.1f%%  %02d:%02d  %.0f/s  ",
                     frac * 100.0, mm, ss, rate);
        std::fflush(stderr);
    }

    void finish() {
        update(total_);
        std::fprintf(stderr, "\n");
    }

private:
    uint64_t   total_;
    std::string label_;
    int         width_;
    std::chrono::steady_clock::time_point start_;
};

} // namespace bbpe
