#pragma once
#include <array>
#include <cstdint>
#include <cstring>

namespace tt_emule {

// 32x32 f32 tile = 4096 bytes
class Tile {
public:
    static constexpr size_t ROWS = 32;
    static constexpr size_t COLS = 32;
    static constexpr size_t NUM_ELEMENTS = ROWS * COLS;
    static constexpr size_t SIZE_BYTES = NUM_ELEMENTS * sizeof(float);

    Tile() { data_.fill(0.0f); }
    explicit Tile(float val) { data_.fill(val); }

    float& operator()(size_t row, size_t col) {
        return data_[row * COLS + col];
    }
    const float& operator()(size_t row, size_t col) const {
        return data_[row * COLS + col];
    }

    Tile operator+(const Tile& other) const {
        Tile result;
        for (size_t i = 0; i < NUM_ELEMENTS; ++i)
            result.data_[i] = data_[i] + other.data_[i];
        return result;
    }

    static constexpr size_t size_bytes() { return SIZE_BYTES; }

    uint8_t* bytes() { return reinterpret_cast<uint8_t*>(data_.data()); }
    const uint8_t* bytes() const { return reinterpret_cast<const uint8_t*>(data_.data()); }

    std::array<float, NUM_ELEMENTS>& raw() { return data_; }
    const std::array<float, NUM_ELEMENTS>& raw() const { return data_; }

private:
    std::array<float, NUM_ELEMENTS> data_;
};

} // namespace tt_emule
