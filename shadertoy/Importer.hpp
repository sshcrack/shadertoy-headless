/*
    SPDX-License-Identifier: Apache-2.0
    Copyright 2023-2026 Yingwei Zheng and contributors
*/
#pragma once

#include "shadertoy/Result.hpp"
#include "shadertoy/STTF.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

SHADERTOY_NAMESPACE_BEGIN

Result<ShaderDocument> makeImageShader(std::string name, std::string source, std::optional<uint32_t> audioChannel = std::nullopt);
Result<ShaderDocument> importFromShaderToy(std::string_view shaderUrlOrId);
Result<ShaderDocument> importFromShaderToyResponse(std::string_view shaderId, std::string_view responseBody);

SHADERTOY_NAMESPACE_END
