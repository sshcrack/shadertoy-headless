/*
    SPDX-License-Identifier: Apache-2.0
    Copyright 2023-2026 Yingwei Zheng and contributors
*/
#pragma once

#include "shadertoy/Config.hpp"
#include "shadertoy/Error.hpp"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "shadertoy/SuppressWarningPush.hpp"
#include <gsl/gsl>
#include "shadertoy/SuppressWarningPop.hpp"

SHADERTOY_NAMESPACE_BEGIN

template <typename F>
auto scopeExit(F&& f) {
    return gsl::finally(std::forward<F>(f));
}

template <typename F>
auto scopeFail(F&& f) {
    return gsl::finally([func = std::forward<F>(f)]() mutable {
        if(std::uncaught_exceptions())
            func();
    });
}

using Clock = std::chrono::steady_clock;

inline std::size_t checkedSizeProduct(const std::initializer_list<std::size_t> factors, const std::string_view label) {
    std::size_t result = 1;
    for(const auto factor : factors) {
        if(factor != 0 && result > std::numeric_limits<std::size_t>::max() / factor)
            throw Error(std::string(label) + " size overflows addressable memory");
        result *= factor;
    }
    return result;
}

inline std::size_t checkedSizeSum(const std::initializer_list<std::size_t> terms, const std::string_view label) {
    std::size_t result = 0;
    for(const auto term : terms) {
        if(result > std::numeric_limits<std::size_t>::max() - term)
            throw Error(std::string(label) + " size overflows addressable memory");
        result += term;
    }
    return result;
}

#ifdef NDEBUG
#if defined(__cpp_lib_unreachable)
#define SHADERTOY_UNREACHABLE() std::unreachable()
#elif defined(__GNUC__)
#define SHADERTOY_UNREACHABLE() __builtin_unreachable()
#elif defined(_MSC_VER)
#define SHADERTOY_UNREACHABLE() __assume(false)
#else
#define SHADERTOY_UNREACHABLE() ::ShaderToy::reportFatalError("unreachable")
#endif
#else
#define SHADERTOY_UNREACHABLE() ::ShaderToy::reportFatalError("unreachable")
#endif

[[noreturn]] inline void reportFatalError(const std::string_view error) {
    throw Error(std::string(error));
}

[[noreturn]] inline void reportNotImplemented() {
    reportFatalError("Not implemented feature");
}

SHADERTOY_NAMESPACE_END
