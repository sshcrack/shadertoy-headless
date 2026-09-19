/*
    SPDX-License-Identifier: Apache-2.0
    Copyright 2023-2026 Yingwei Zheng and contributors
*/
#pragma once

#include "shadertoy/Config.hpp"

#include <array>
#include <cstdint>

SHADERTOY_NAMESPACE_BEGIN

struct Vec2 final {
    float x{};
    float y{};
};

struct Vec4 final {
    float x{};
    float y{};
    float z{};
    float w{};
};

struct RenderRegion final {
    Vec2 framebufferSize;
    Vec2 clipMin;
    Vec2 clipMax;
    Vec2 canvasSize;
};

struct MouseInput final {
    float x{};
    float y{};
    bool down{};
    bool clicked{};
};

class KeyboardInput final {
public:
    static constexpr std::size_t KeyCount = 256;
    static constexpr std::size_t Rows = 3;

    void setKey(const std::uint8_t key, const bool down, const bool pressed) noexcept {
        constexpr std::uint32_t mask = 0xffffffffU;
        at(key, 0) = down ? mask : 0U;
        at(key, 1) = pressed ? mask : 0U;
        if(pressed)
            at(key, 2) ^= mask;
    }

    void clearTransient() noexcept {
        for(std::size_t key = 0; key < KeyCount; ++key)
            at(key, 1) = 0U;
    }

    [[nodiscard]] const std::array<std::uint32_t, KeyCount * Rows>& pixels() const noexcept {
        return mPixels;
    }

private:
    [[nodiscard]] std::uint32_t& at(const std::size_t key, const std::size_t row) noexcept {
        return mPixels[key + row * KeyCount];
    }

    std::array<std::uint32_t, KeyCount * Rows> mPixels{};
};

SHADERTOY_NAMESPACE_END
