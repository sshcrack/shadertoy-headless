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
