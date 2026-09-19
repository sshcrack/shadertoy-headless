/*
    SPDX-License-Identifier: Apache-2.0
    Copyright 2023-2026 Yingwei Zheng and contributors
*/
#pragma once

#include "shadertoy/Backend.hpp"
#include "shadertoy/STTF.hpp"

#include <memory>

SHADERTOY_NAMESPACE_BEGIN

/// Compile a GUI-independent shader document into an OpenGL pipeline.
///
/// The caller must have a current OpenGL context for the duration of compilation
/// and for every subsequent render using the returned pipeline.
std::unique_ptr<Pipeline> compilePipeline(const ShaderDocument& document);

SHADERTOY_NAMESPACE_END
