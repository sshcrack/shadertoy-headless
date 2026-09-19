/*
    SPDX-License-Identifier: Apache-2.0
    Copyright 2023-2026 Yingwei Zheng and contributors
*/
#pragma once

#include "shadertoy/Error.hpp"

#include <expected>

SHADERTOY_NAMESPACE_BEGIN

template <typename T>
using Result = std::expected<T, Error>;

SHADERTOY_NAMESPACE_END
