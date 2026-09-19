/*
    SPDX-License-Identifier: Apache-2.0
    Copyright 2023-2026 Yingwei Zheng and contributors
*/
#pragma once

#include "shadertoy/Config.hpp"

#include <stdexcept>
#include <string>
#include <utility>

SHADERTOY_NAMESPACE_BEGIN

class Error final : public std::runtime_error {
public:
    explicit Error(std::string message) : std::runtime_error(std::move(message)) {}
};

SHADERTOY_NAMESPACE_END
