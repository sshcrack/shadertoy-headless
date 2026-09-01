/*
    SPDX-License-Identifier: Apache-2.0
    Copyright 2023-2026 Yingwei Zheng and contributors
*/

#pragma once

#include "shadertoy/Config.hpp"

#include <cstdint>
#include <vector>

SHADERTOY_NAMESPACE_BEGIN

/// Host-provided music analysis for audio-reactive shaders.
///
/// `spectrum` is expected to contain normalized 0..1 magnitudes ordered from
/// low to high frequencies. `waveform` is expected to contain -1..1 samples.
/// The renderer resamples both to ShaderToy's conventional 512x2 music texture:
/// row 0 = spectrum, row 1 = waveform remapped to 0..1.
///
/// The semantic fields are renderer-agnostic and are exposed to GLSL through
/// the iAudio* aliases declared by the renderer header.
struct AudioInput final {
    static constexpr uint32_t TextureWidth = 512;
    static constexpr uint32_t TextureHeight = 2;

    bool available = false;
    bool silence = true;
    float sampleRate = 0.0f;

    std::vector<float> spectrum;
    std::vector<float> waveform;

    float loudness = 0.0f;
    float bass = 0.0f;
    float mid = 0.0f;
    float treble = 0.0f;

    float onset = 0.0f;
    float kick = 0.0f;
    float snare = 0.0f;
    float hihat = 0.0f;

    float bpm = 0.0f;
    float beatPhase = 0.0f;
    float beatConfidence = 0.0f;
    float beatStrength = 0.0f;

    float stereoWidth = 0.0f;
    float stereoBalance = 0.0f;
    float stereoCorrelation = 0.0f;
    float energyTrend = 0.0f;

    float drop = 0.0f;
    float sectionChange = 0.0f;
    float spectralCentroid = 0.0f;
    float spectralFlux = 0.0f;
};

SHADERTOY_NAMESPACE_END
