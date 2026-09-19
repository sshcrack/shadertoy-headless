/*
    SPDX-License-Identifier: Apache-2.0
    Copyright 2023-2025 Yingwei Zheng
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include "shadertoy/Backend.hpp"
#include "shadertoy/Support.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>

#include "shadertoy/SuppressWarningPush.hpp"

#include <glad/glad.h>

#include "shadertoy/SuppressWarningPop.hpp"

#include <cmath>

SHADERTOY_NAMESPACE_BEGIN

namespace {
    uint32_t checkedPixelDimension(const float value, const char* label) {
        if(!std::isfinite(value) || value < 1.0f ||
           static_cast<double>(value) > static_cast<double>(std::numeric_limits<GLsizei>::max()) || std::floor(value) != value)
            throw Error(std::string(label) + " must be a positive integer within the OpenGL dimension range");
        return static_cast<uint32_t>(value);
    }

    void validateFramebufferDimensions(const uint32_t width, const uint32_t height) {
        if(width == 0 || height == 0 || width > static_cast<uint32_t>(std::numeric_limits<GLsizei>::max()) ||
           height > static_cast<uint32_t>(std::numeric_limits<GLsizei>::max()))
            throw Error("Framebuffer dimensions are outside the OpenGL dimension range");
    }
}  // namespace

static const char* const shaderVersionDirective = "#version 410 core\n";
static const char* const shaderCubeMapDef = "#define INTERFACE_SHADERTOY_CUBE_MAP\n";
static const char* const shaderVertexSrc = R"(
layout (location = 0) in vec2 pos;
layout (location = 1) in vec2 texCoord;
#ifdef INTERFACE_SHADERTOY_CUBE_MAP
layout (location = 2) in vec3 point;
#endif

layout (location = 0) out vec2 f_fragCoord;
#ifdef INTERFACE_SHADERTOY_CUBE_MAP
layout (location = 1) out vec3 f_point;
#endif

void main() {
    gl_Position = vec4(pos, 0.0f, 1.0f);
    f_fragCoord = texCoord;
#ifdef INTERFACE_SHADERTOY_CUBE_MAP
    f_point = point;
#endif
}

)";

static const char* const shaderPixelHeader = R"(
layout (location = 0) in vec2 f_fragCoord;
#ifdef INTERFACE_SHADERTOY_CUBE_MAP
layout (location = 1) in vec3 f_point;
#endif

layout (location = 0) out vec4 out_frag_color;

uniform vec3      iResolution;           // viewport resolution (in pixels)
uniform float     iTime;                 // shader playback time (in seconds)
uniform float     iTimeDelta;            // render time (in seconds)
uniform float     iFrameRate;            // shader frame rate
uniform int       iFrame;                // shader playback frame
uniform vec4      iMouse;                // mouse pixel coords. xy: current (if MLB down), zw: click
uniform vec4      iDate;                 // Year, month, day, time in seconds in .xyzw
uniform vec3 iChannelResolution[4];

// Host-provided semantic music analysis. The packed vec4 uniforms keep the
// renderer interface small; the aliases below are the shader-facing contract.
uniform vec4 iMusicBands;                // loudness, bass, mid, treble
uniform vec4 iMusicHits;                 // onset, kick, snare, hihat
uniform vec4 iMusicBeat;                 // bpm, phase, confidence, strength
uniform vec4 iMusicStereo;               // width, balance, correlation, energy trend
uniform vec4 iMusicStructure;            // drop, section change, spectral centroid, spectral flux
uniform vec4 iMusicMeta;                 // available, silence, sample rate, reserved

#define iAudioLoudness          (iMusicBands.x)
#define iAudioBass              (iMusicBands.y)
#define iAudioMid               (iMusicBands.z)
#define iAudioTreble            (iMusicBands.w)
#define iAudioOnset             (iMusicHits.x)
#define iAudioKick              (iMusicHits.y)
#define iAudioSnare             (iMusicHits.z)
#define iAudioHihat             (iMusicHits.w)
#define iAudioBpm               (iMusicBeat.x)
#define iAudioBeatPhase         (iMusicBeat.y)
#define iAudioBeatConfidence    (iMusicBeat.z)
#define iAudioBeatStrength      (iMusicBeat.w)
#define iAudioStereoWidth       (iMusicStereo.x)
#define iAudioStereoBalance     (iMusicStereo.y)
#define iAudioStereoCorrelation (iMusicStereo.z)
#define iAudioEnergyTrend       (iMusicStereo.w)
#define iAudioDrop              (iMusicStructure.x)
#define iAudioSectionChange     (iMusicStructure.y)
#define iAudioSpectralCentroid  (iMusicStructure.z)
#define iAudioSpectralFlux      (iMusicStructure.w)
#define iAudioAvailable         (iMusicMeta.x)
#define iAudioSilence           (iMusicMeta.y)
#define iAudioSampleRate        (iMusicMeta.z)

float sampleAudioSpectrum(sampler2D channel, float x) {
    return texture(channel, vec2(clamp(x, 0.0, 1.0), 0.25)).r;
}
float sampleAudioWaveform(sampler2D channel, float x) {
    return texture(channel, vec2(clamp(x, 0.0, 1.0), 0.75)).r * 2.0 - 1.0;
}

#define char char_
)";

static const char* const shaderPixelFooter = R"(
void main() {
#ifdef SHADERTOY_CLAMP_OUTPUT
    out_frag_color = vec4(0.0f, 0.0f, 0.0f, 1.0f);
#endif
    vec4 output_color = vec4(1e20f);
#ifndef INTERFACE_SHADERTOY_CUBE_MAP
    mainImage(output_color, gl_FragCoord.xy);
#else
    mainCubemap(output_color, gl_FragCoord.xy, vec3(0.0), normalize(f_point));
#endif
#ifdef SHADERTOY_CLAMP_OUTPUT
    out_frag_color = vec4(clamp(output_color.xyz, vec3(0.0f), vec3(1.0f)), 1.0f);
#else
    out_frag_color = output_color;
#endif
}
)";

struct Vertex final {
    Vec2 pos;
    Vec2 coord;
};

using Vec3 = std::array<float, 3>;

constexpr Vec3 cubeMapVertexPos[8] = { { -1.0f, -1.0f, -1.0f }, { -1.0f, -1.0f, 1.0f },  //
                                       { -1.0f, 1.0f, -1.0f },  { -1.0f, 1.0f, 1.0f },   //
                                       { 1.0f, -1.0f, -1.0f },  { 1.0f, -1.0f, 1.0f },   //
                                       { 1.0f, 1.0f, -1.0f },   { 1.0f, 1.0f, 1.0f } };
// OpenGL cubemap face order: +X, -X, +Y, -Y, +Z, -Z (0,1,2,3,4,5)
// Vertices are ordered: left-bottom, left-top, right-top, right-bottom
// NOTE: The render code swaps +Y (idx=2) and -Y (idx=3) for CubeMapFlippedY
// So we need to design indices accounting for this swap
constexpr uint32_t cubeMapVertexIndex[6][4] = {
    { 5, 7, 6, 4 },  // +X (right face)
    { 0, 2, 3, 1 },  // -X (left face)
    { 0, 1, 5, 4 },  // +Y (top face)
    { 3, 2, 6, 7 },  // -Y (bottom face)
    { 1, 3, 7, 5 },  // +Z (back face)
    { 4, 6, 2, 0 }   // -Z (front face)
};

struct VertexCubeMap final {  // NOLINT(cppcoreguidelines-pro-type-member-init)
    Vec2 pos;
    Vec2 coord;
    Vec3 point;
};

static void checkShaderCompileError(const GLuint shader, const std::string_view type) {
    GLint success;
    std::vector<GLchar> buffer;
    GLint size;
    if(type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if(!success) {
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &size);
            buffer.resize(static_cast<size_t>(size));
            glGetShaderInfoLog(shader, static_cast<GLsizei>(buffer.size()), nullptr, buffer.data());
            throw std::runtime_error(buffer.data());
        }
    } else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if(!success) {
            glGetProgramiv(shader, GL_INFO_LOG_LENGTH, &size);
            buffer.resize(static_cast<size_t>(size));
            glGetProgramInfoLog(shader, static_cast<GLsizei>(buffer.size()), nullptr, buffer.data());
            throw std::runtime_error(buffer.data());
        }
    }
}

class GLFrameBuffer final : public FrameBuffer {
    GLuint mFBO{};
    GLuint mTexture{};
    uint32_t mWidth = 0, mHeight = 0;

public:
    GLFrameBuffer() {
        glGenFramebuffers(1, &mFBO);
        glGenTextures(1, &mTexture);
        glBindFramebuffer(GL_FRAMEBUFFER, mFBO);
        glBindTexture(GL_TEXTURE_2D, mTexture);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mTexture, 0);
        glBindTexture(GL_TEXTURE_2D, GL_NONE);
        glBindFramebuffer(GL_FRAMEBUFFER, GL_NONE);
    }
    GLFrameBuffer(const GLFrameBuffer&) = delete;
    GLFrameBuffer(GLFrameBuffer&&) = delete;
    GLFrameBuffer& operator=(const GLFrameBuffer&) = delete;
    GLFrameBuffer& operator=(GLFrameBuffer&&) = delete;
    ~GLFrameBuffer() override {
        glDeleteFramebuffers(1, &mFBO);
        glDeleteTextures(1, &mTexture);
    }
    void bind(const uint32_t width, const uint32_t height) override {
        validateFramebufferDimensions(width, height);
        if(width != mWidth || height != mHeight) {
            glBindTexture(GL_TEXTURE_2D, mTexture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, static_cast<GLsizei>(width), static_cast<GLsizei>(height), 0, GL_RGBA,
                         GL_FLOAT, nullptr);
            glBindTexture(GL_TEXTURE_2D, GL_NONE);
            mWidth = width;
            mHeight = height;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, mFBO);
        assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    }
    void unbind() override {
        glBindFramebuffer(GL_FRAMEBUFFER, GL_NONE);
    }

    [[nodiscard]] uintptr_t getTexture() const override {
        return mTexture;
    }
    [[nodiscard]] std::vector<uint8_t> readRgb() override {
        if(mWidth == 0 || mHeight == 0)
            throw Error("Framebuffer has not been rendered yet");
        GLint previousFramebuffer = 0;
        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &previousFramebuffer);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, mFBO);
        std::vector<uint8_t> result(checkedSizeProduct({ mWidth, mHeight, 3U }, "Framebuffer RGB readback"));
        glReadPixels(0, 0, static_cast<GLsizei>(mWidth), static_cast<GLsizei>(mHeight), GL_RGB, GL_UNSIGNED_BYTE, result.data());
        glBindFramebuffer(GL_READ_FRAMEBUFFER, static_cast<GLuint>(previousFramebuffer));
        return result;
    }
    [[nodiscard]] std::vector<float> readRgba32f() override {
        if(mWidth == 0 || mHeight == 0)
            throw Error("Framebuffer has not been rendered yet");
        GLint previousFramebuffer = 0;
        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &previousFramebuffer);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, mFBO);
        std::vector<float> result(checkedSizeProduct({ mWidth, mHeight, 4U }, "Framebuffer RGBA readback"));
        glReadPixels(0, 0, static_cast<GLsizei>(mWidth), static_cast<GLsizei>(mHeight), GL_RGBA, GL_FLOAT, result.data());
        glBindFramebuffer(GL_READ_FRAMEBUFFER, static_cast<GLuint>(previousFramebuffer));
        return result;
    }
    void writeRgba8(const uint32_t width, const uint32_t height, const uint8_t* data) override {
        validateFramebufferDimensions(width, height);
        glBindTexture(GL_TEXTURE_2D, mTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, static_cast<GLsizei>(width), static_cast<GLsizei>(height), 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, data);
        glBindTexture(GL_TEXTURE_2D, GL_NONE);
        mWidth = width;
        mHeight = height;
    }
    void writeRgba32f(const uint32_t width, const uint32_t height, const float* data) override {
        validateFramebufferDimensions(width, height);
        glBindTexture(GL_TEXTURE_2D, mTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, static_cast<GLsizei>(width), static_cast<GLsizei>(height), 0, GL_RGBA,
                     GL_FLOAT, data);
        glBindTexture(GL_TEXTURE_2D, GL_NONE);
        mWidth = width;
        mHeight = height;
    }
};

static constexpr uint32_t cubeMapRenderTargetSize = 1024;
class GLCubeMapRenderTarget final {
    GLuint mTex{};

public:
    GLCubeMapRenderTarget() {
        glGenTextures(1, &mTex);
        glBindTexture(GL_TEXTURE_CUBE_MAP, mTex);
        for(int32_t idx = 0; idx < 6; ++idx) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + idx, 0, GL_RGBA16F, static_cast<GLsizei>(cubeMapRenderTargetSize),
                         static_cast<GLsizei>(cubeMapRenderTargetSize), 0, GL_RGBA, GL_HALF_FLOAT, nullptr);
        }
        glBindTexture(GL_TEXTURE_CUBE_MAP, GL_NONE);
    }
    GLCubeMapRenderTarget(const GLCubeMapRenderTarget&) = delete;
    GLCubeMapRenderTarget(GLCubeMapRenderTarget&&) = delete;
    GLCubeMapRenderTarget& operator=(const GLCubeMapRenderTarget&) = delete;
    GLCubeMapRenderTarget& operator=(GLCubeMapRenderTarget&&) = delete;
    ~GLCubeMapRenderTarget() {
        glDeleteTextures(1, &mTex);
    }
    [[nodiscard]] GLuint getTexture() const {
        return mTex;
    }
};

class GLCubeMapFrameBuffer final : public FrameBuffer {
    GLuint mFBO{};
    GLuint mTexture{};

public:
    GLCubeMapFrameBuffer(GLuint texture, uint32_t idx) : mTexture{ texture } {
        glGenFramebuffers(1, &mFBO);
        glBindFramebuffer(GL_FRAMEBUFFER, mFBO);
        glBindTexture(GL_TEXTURE_CUBE_MAP, mTexture);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + idx, mTexture, 0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, GL_NONE);
        glBindFramebuffer(GL_FRAMEBUFFER, GL_NONE);
    }
    GLCubeMapFrameBuffer(const GLCubeMapFrameBuffer&) = delete;
    GLCubeMapFrameBuffer(GLCubeMapFrameBuffer&&) = delete;
    GLCubeMapFrameBuffer& operator=(const GLCubeMapFrameBuffer&) = delete;
    GLCubeMapFrameBuffer& operator=(GLCubeMapFrameBuffer&&) = delete;
    ~GLCubeMapFrameBuffer() override {
        glDeleteFramebuffers(1, &mFBO);
    }
    void bind(const uint32_t, const uint32_t) override {
        glBindFramebuffer(GL_FRAMEBUFFER, mFBO);
        assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    }
    void unbind() override {
        glBindFramebuffer(GL_FRAMEBUFFER, GL_NONE);
    }

    [[nodiscard]] uintptr_t getTexture() const override {
        return mTexture;
    }
    [[nodiscard]] std::vector<uint8_t> readRgb() override {
        GLint previousFramebuffer = 0;
        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &previousFramebuffer);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, mFBO);
        std::vector<uint8_t> result(
            checkedSizeProduct({ cubeMapRenderTargetSize, cubeMapRenderTargetSize, 3U }, "Cubemap RGB readback"));
        glReadPixels(0, 0, static_cast<GLsizei>(cubeMapRenderTargetSize), static_cast<GLsizei>(cubeMapRenderTargetSize), GL_RGB,
                     GL_UNSIGNED_BYTE, result.data());
        glBindFramebuffer(GL_READ_FRAMEBUFFER, static_cast<GLuint>(previousFramebuffer));
        return result;
    }
    [[nodiscard]] std::vector<float> readRgba32f() override {
        throw Error("Cubemap pass state snapshots are not supported");
    }
    void writeRgba8(const uint32_t, const uint32_t, const uint8_t*) override {
        throw Error("Cubemap pass overrides are not supported");
    }
    void writeRgba32f(const uint32_t, const uint32_t, const float*) override {
        throw Error("Cubemap pass state restores are not supported");
    }
};

class RenderPass final {
    std::string mName;
    GLuint mProgram;
    std::vector<DoubleBufferedFB> mBuffers;
    NodeType mType;
    GLint mLocationResolution;
    GLint mLocationTime;
    GLint mLocationTimeDelta;
    GLint mLocationFrameRate;
    GLint mLocationFrame;
    GLint mLocationMouse;
    GLint mLocationDate;
    GLint mLocationMusicBands;
    GLint mLocationMusicHits;
    GLint mLocationMusicBeat;
    GLint mLocationMusicStereo;
    GLint mLocationMusicStructure;
    GLint mLocationMusicMeta;
    GLint mLocationChannel[4]{};
    GLint mLocationChannelResolution[4]{};
    std::vector<Channel> mChannels;

public:
    RenderPass(std::string name, const std::string& src, NodeType type, std::vector<DoubleBufferedFB> buffer,
               std::vector<Channel> channels, bool clampOutput)
        : mName{ std::move(name) }, mBuffers{ std::move(buffer) }, mType{ type }, mChannels{ std::move(channels) } {
        std::string vertexSrc = shaderVersionDirective;
        std::string pixelSrc = shaderVersionDirective;
        if(type == NodeType::CubeMap) {
            vertexSrc += shaderCubeMapDef;
            pixelSrc += shaderCubeMapDef;
        }

        vertexSrc += shaderVertexSrc;
        pixelSrc += shaderPixelHeader;
        for(auto& channel : mChannels) {
            pixelSrc += "uniform sampler";
            pixelSrc += channel.tex.type == TexType::CubeMap ? "Cube" : channel.tex.type == TexType::Tex2D ? "2D" : "3D";
            pixelSrc += " iChannel";
            pixelSrc += static_cast<char>(static_cast<uint32_t>('0') + channel.slot);
            pixelSrc += ";\n";
        }
        if(clampOutput)
            pixelSrc += "#define SHADERTOY_CLAMP_OUTPUT\n";
        pixelSrc += "#line 1\n";
        pixelSrc += src;
        pixelSrc += shaderPixelFooter;

        const auto vertexSrcData = vertexSrc.c_str();
        const auto pixelSrcData = pixelSrc.c_str();

        // std::cout << "---- Vertex Shader ----" << std::endl;
        // std::cout << vertexSrc << std::endl;
        // std::cout << "------------------------" << std::endl;

        const auto shaderVertex = glCreateShader(GL_VERTEX_SHADER);
        auto vertGuard = scopeExit([&] { glDeleteShader(shaderVertex); });
        glShaderSource(shaderVertex, 1, &vertexSrcData, nullptr);
        glCompileShader(shaderVertex);
        checkShaderCompileError(shaderVertex, "VERTEX");

        // std::cout << "---- Pixel Shader ----" << std::endl;
        // std::cout << pixelSrc << std::endl;
        // std::cout << "-----------------------" << std::endl;

        const auto shaderPixel = glCreateShader(GL_FRAGMENT_SHADER);
        auto pixelGuard = scopeExit([&] { glDeleteShader(shaderPixel); });
        glShaderSource(shaderPixel, 1, &pixelSrcData, nullptr);
        glCompileShader(shaderPixel);
        checkShaderCompileError(shaderPixel, "PIXEL");

        mProgram = glCreateProgram();
        auto programGuard = scopeFail([&] { glDeleteProgram(mProgram); });
        glAttachShader(mProgram, shaderVertex);
        auto vertBindGuard = scopeExit([&] { glDetachShader(mProgram, shaderVertex); });
        glAttachShader(mProgram, shaderPixel);
        auto pixelBindGuard = scopeExit([&] { glDetachShader(mProgram, shaderPixel); });
        glLinkProgram(mProgram);
        checkShaderCompileError(mProgram, "PROGRAM");

        auto& mLocationChannel0 = mLocationChannel[0];
        auto& mLocationChannel1 = mLocationChannel[1];
        auto& mLocationChannel2 = mLocationChannel[2];
        auto& mLocationChannel3 = mLocationChannel[3];
#define SHADERTOY_GET_UNIFORM_LOCATION(NAME) mLocation##NAME = glGetUniformLocation(mProgram, "i" #NAME)
        SHADERTOY_GET_UNIFORM_LOCATION(Resolution);
        SHADERTOY_GET_UNIFORM_LOCATION(Time);
        SHADERTOY_GET_UNIFORM_LOCATION(TimeDelta);
        SHADERTOY_GET_UNIFORM_LOCATION(FrameRate);
        SHADERTOY_GET_UNIFORM_LOCATION(Frame);
        SHADERTOY_GET_UNIFORM_LOCATION(Mouse);
        SHADERTOY_GET_UNIFORM_LOCATION(Date);
        SHADERTOY_GET_UNIFORM_LOCATION(MusicBands);
        SHADERTOY_GET_UNIFORM_LOCATION(MusicHits);
        SHADERTOY_GET_UNIFORM_LOCATION(MusicBeat);
        SHADERTOY_GET_UNIFORM_LOCATION(MusicStereo);
        SHADERTOY_GET_UNIFORM_LOCATION(MusicStructure);
        SHADERTOY_GET_UNIFORM_LOCATION(MusicMeta);
        SHADERTOY_GET_UNIFORM_LOCATION(Channel0);
        SHADERTOY_GET_UNIFORM_LOCATION(Channel1);
        SHADERTOY_GET_UNIFORM_LOCATION(Channel2);
        SHADERTOY_GET_UNIFORM_LOCATION(Channel3);
        SHADERTOY_GET_UNIFORM_LOCATION(ChannelResolution[0]);
        SHADERTOY_GET_UNIFORM_LOCATION(ChannelResolution[1]);
        SHADERTOY_GET_UNIFORM_LOCATION(ChannelResolution[2]);
        SHADERTOY_GET_UNIFORM_LOCATION(ChannelResolution[3]);
#undef SHADERTOY_GET_UNIFORM_LOCATION
    }
    RenderPass(const RenderPass&) = delete;
    RenderPass(RenderPass&&) = delete;
    RenderPass& operator=(const RenderPass&) = delete;
    RenderPass& operator=(RenderPass&&) = delete;
    ~RenderPass() {
        glDeleteProgram(mProgram);
    }
    [[nodiscard]] NodeType getType() const noexcept {
        return mType;
    }
    [[nodiscard]] std::string_view getName() const noexcept {
        return mName;
    }
    [[nodiscard]] bool hasOffscreenTarget() const noexcept {
        return !mBuffers.empty() && mBuffers.front().t1 != nullptr;
    }
    [[nodiscard]] std::vector<uint8_t> readRgb() {
        if(mType != NodeType::Image)
            throw Error("Only image/buffer passes can be read as RGB");
        if(!hasOffscreenTarget())
            throw Error("Final image pass is rendered to the caller framebuffer");
        return mBuffers.front().t1->readRgb();
    }
    [[nodiscard]] std::vector<float> readRgba32f() {
        if(mType != NodeType::Image)
            throw Error("Only image/buffer passes can be snapshotted as RGBA32F");
        if(!hasOffscreenTarget())
            throw Error("The final image pass has no persistent buffer state");
        return mBuffers.front().t1->readRgba32f();
    }
    void overrideRgba8(const uint32_t width, const uint32_t height, const uint8_t* data) {
        if(mType != NodeType::Image)
            throw Error("Only image/buffer passes can be overridden with a 2D image");
        if(!hasOffscreenTarget())
            throw Error("The final image pass cannot be used as a persistent buffer override");
        auto* first = mBuffers.front().t1;
        auto* second = mBuffers.front().t2;
        first->writeRgba8(width, height, data);
        if(second && second != first)
            second->writeRgba8(width, height, data);
    }
    void restoreRgba32f(const uint32_t width, const uint32_t height, const float* data) {
        if(mType != NodeType::Image)
            throw Error("Only image/buffer passes can restore RGBA32F state");
        if(!hasOffscreenTarget())
            throw Error("The final image pass has no persistent buffer state");
        auto* first = mBuffers.front().t1;
        auto* second = mBuffers.front().t2;
        first->writeRgba32f(width, height, data);
        if(second && second != first)
            second->writeRgba32f(width, height, data);
    }
    void render(const Vec2 frameBufferSize, const Vec2 clipMin, const Vec2 clipMax, const Vec2 canvasSize,
                const ShaderToyUniform& uniform, const GLuint vao, const GLuint vbo) {
        glDisable(GL_BLEND);
        constexpr Vec2 cubeMapSize{ static_cast<float>(cubeMapRenderTargetSize), static_cast<float>(cubeMapRenderTargetSize) };
        const auto screenBase = clipMin;
        const auto screenSize = Vec2{ clipMax.x - clipMin.x, clipMax.y - clipMin.y };

        for(uint32_t idx = 0; idx < mBuffers.size(); ++idx) {
            const auto buffer = mBuffers[idx].get();
            Vec2 size, base, fbSize, uniformSize;
            if(buffer) {
                base = { 0, 0 };
                size = mType == NodeType::CubeMap ? cubeMapSize : screenSize;
                fbSize = size;
                uniformSize = mType == NodeType::CubeMap ? cubeMapSize : canvasSize;
                glViewport(0, 0, static_cast<GLsizei>(size.x), static_cast<GLsizei>(size.y));
                glDisable(GL_SCISSOR_TEST);
                buffer->bind(static_cast<uint32_t>(size.x), static_cast<uint32_t>(size.y));
            } else {
                glViewport(0, 0, static_cast<GLsizei>(frameBufferSize.x), static_cast<GLsizei>(frameBufferSize.y));
                glEnable(GL_SCISSOR_TEST);
                glScissor(static_cast<GLint>(clipMin.x), static_cast<GLint>(frameBufferSize.y - clipMax.y),
                          static_cast<GLint>(clipMax.x - clipMin.x), static_cast<GLint>(clipMax.y - clipMin.y));
                base = screenBase;
                size = screenSize;
                fbSize = frameBufferSize;
                uniformSize = canvasSize;
            }
            glUseProgram(mProgram);
            // update vertex array
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBindVertexArray(vao);
            if(mType == NodeType::Image) {
                std::array vertices{
                    Vertex{ Vec2{ base.x, base.y + size.y }, Vec2{ 0.0, 0.0 } },                      // left-bottom
                    Vertex{ Vec2{ base.x, base.y }, Vec2{ 0.0, uniformSize.y } },                     // left-top
                    Vertex{ Vec2{ base.x + size.x, base.y }, Vec2{ uniformSize.x, uniformSize.y } },  // right-top
                    Vertex{ Vec2{ base.x + size.x, base.y + size.y }, Vec2{ uniformSize.x, 0.0 } },   // right-bottom
                };
                for(auto& [pos, coord] : vertices) {
                    pos.x = pos.x / fbSize.x * 2.0f - 1.0f;
                    pos.y = 1.0f - pos.y / fbSize.y * 2.0f;
                }
                glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(Vertex), vertices.data(), GL_STREAM_DRAW);
            } else {
                // For flipped Y cubemaps, swap +Y (idx=2) and -Y (idx=3) face geometry
                // AND flip Y coordinates to match ShaderToy's vflip behavior
                std::array vertices{
                    VertexCubeMap{ Vec2{ base.x, base.y + size.y }, Vec2{ 0.0, 0.0 },
                                   cubeMapVertexPos[cubeMapVertexIndex[idx][0]] },  // left-bottom
                    VertexCubeMap{ Vec2{ base.x, base.y }, Vec2{ 0.0, uniformSize.y },
                                   cubeMapVertexPos[cubeMapVertexIndex[idx][1]] },  // left-top
                    VertexCubeMap{ Vec2{ base.x + size.x, base.y }, Vec2{ uniformSize.x, uniformSize.y },
                                   cubeMapVertexPos[cubeMapVertexIndex[idx][2]] },  // right-top
                    VertexCubeMap{ Vec2{ base.x + size.x, base.y + size.y }, Vec2{ uniformSize.x, 0.0 },
                                   cubeMapVertexPos[cubeMapVertexIndex[idx][3]] },  // right-bottom
                };

                // For Y-flipped cubemaps, also flip Y coordinates of all faces
                // to match ShaderToy's UNPACK_FLIP_Y_WEBGL behavior
                // CubeMaps are always Y-flipped
                for(auto& [pos, coord, point] : vertices) {
                    // Flip the 3D point's Y coordinate
                    point[1] = -point[1];
                }

                for(auto& [pos, coord, point] : vertices) {
                    pos.x = pos.x / fbSize.x * 2.0f - 1.0f;
                    pos.y = 1.0f - pos.y / fbSize.y * 2.0f;
                }
                glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(VertexCubeMap), vertices.data(), GL_STREAM_DRAW);
            }

            // update texture
            for(auto& channel : mChannels) {
                if(mLocationChannelResolution[channel.slot] == -1)
                    continue;
                if(channel.tex.type != TexType::Tex3D) {
                    const auto texSize = channel.size.value_or(channel.tex.type == TexType::CubeMap ? cubeMapSize : size);
                    glUniform3f(mLocationChannelResolution[channel.slot], texSize.x, texSize.y, 1.0f);
                } else {
                    const auto x = channel.size->x;
                    glUniform3f(mLocationChannelResolution[channel.slot], x, x, x);
                }
            }
            for(auto& channel : mChannels) {
                if(mLocationChannel[channel.slot] == -1)
                    continue;
                glUniform1i(mLocationChannel[channel.slot], static_cast<GLint>(channel.slot));
                glActiveTexture(GL_TEXTURE0 + channel.slot);
                const auto type = channel.tex.type == TexType::CubeMap ? GL_TEXTURE_CUBE_MAP :
                    channel.tex.type == TexType::Tex2D                 ? GL_TEXTURE_2D :
                                                                         GL_TEXTURE_3D;
                glBindTexture(type, static_cast<GLuint>(channel.tex.get()));
                // updating
                if(glGetError() != GL_NO_ERROR)
                    continue;
                const GLint wrapMode = [&] {
                    switch(channel.wrapMode) {
                        case Wrap::Clamp:
                            return GL_CLAMP_TO_EDGE;
                        case Wrap::Repeat:
                            return GL_REPEAT;
                    }
                    SHADERTOY_UNREACHABLE();
                }();
                const GLint minFilter = [&] {
                    switch(channel.filter) {
                        case Filter::Mipmap:
                            return GL_LINEAR_MIPMAP_LINEAR;
                        case Filter::Nearest:
                            return GL_NEAREST;
                        case Filter::Linear:
                            return GL_LINEAR;
                    }
                    SHADERTOY_UNREACHABLE();
                }();
                const GLint magFilter = [&] {
                    switch(channel.filter) {
                        case Filter::Nearest:
                            return GL_NEAREST;
                        case Filter::Mipmap:
                            [[fallthrough]];
                        case Filter::Linear:
                            return GL_LINEAR;
                    }
                    SHADERTOY_UNREACHABLE();
                }();
                if(channel.filter == Filter::Mipmap)
                    glGenerateMipmap(type);
                if(channel.tex.type == TexType::Tex3D)
                    glTexParameteri(type, GL_TEXTURE_WRAP_R, wrapMode);

                glTexParameteri(type, GL_TEXTURE_WRAP_S, wrapMode);
                glTexParameteri(type, GL_TEXTURE_WRAP_T, wrapMode);
                glTexParameteri(type, GL_TEXTURE_MIN_FILTER, minFilter);
                glTexParameteri(type, GL_TEXTURE_MAG_FILTER, magFilter);
            }

            // update uniform
            if(mLocationResolution != -1)
                glUniform3f(mLocationResolution, uniformSize.x, uniformSize.y, 0.0f);
            if(mLocationTime != -1)
                glUniform1f(mLocationTime, uniform.time);
            if(mLocationTimeDelta != -1)
                glUniform1f(mLocationTimeDelta, uniform.timeDelta);
            if(mLocationFrameRate != -1)
                glUniform1f(mLocationFrameRate, uniform.frameRate);
            if(mLocationFrame != -1)
                glUniform1i(mLocationFrame, uniform.frame);
            if(mLocationMouse != -1)
                glUniform4f(mLocationMouse, uniform.mouse.x, uniform.mouse.y, uniform.mouse.z, uniform.mouse.w);
            if(mLocationDate != -1)
                glUniform4f(mLocationDate, uniform.date.x, uniform.date.y, uniform.date.z, uniform.date.w);
            if(mLocationMusicBands != -1)
                glUniform4f(mLocationMusicBands, uniform.audioBands.x, uniform.audioBands.y, uniform.audioBands.z,
                            uniform.audioBands.w);
            if(mLocationMusicHits != -1)
                glUniform4f(mLocationMusicHits, uniform.audioHits.x, uniform.audioHits.y, uniform.audioHits.z,
                            uniform.audioHits.w);
            if(mLocationMusicBeat != -1)
                glUniform4f(mLocationMusicBeat, uniform.audioBeat.x, uniform.audioBeat.y, uniform.audioBeat.z,
                            uniform.audioBeat.w);
            if(mLocationMusicStereo != -1)
                glUniform4f(mLocationMusicStereo, uniform.audioStereo.x, uniform.audioStereo.y, uniform.audioStereo.z,
                            uniform.audioStereo.w);
            if(mLocationMusicStructure != -1)
                glUniform4f(mLocationMusicStructure, uniform.audioStructure.x, uniform.audioStructure.y, uniform.audioStructure.z,
                            uniform.audioStructure.w);
            if(mLocationMusicMeta != -1)
                glUniform4f(mLocationMusicMeta, uniform.audioMeta.x, uniform.audioMeta.y, uniform.audioMeta.z,
                            uniform.audioMeta.w);

            glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
            if(buffer)
                buffer->unbind();
        }

        glActiveTexture(GL_TEXTURE0);  // restore
    }
};

class GLTextureObject final : public TextureObject {
    GLuint mTex{};
    Vec2 mSize;

public:
    GLTextureObject(const uint32_t width, const uint32_t height, const uint32_t* data)
        : mSize{ static_cast<float>(width), static_cast<float>(height) } {
        glGenTextures(1, &mTex);
        glBindTexture(GL_TEXTURE_2D, mTex);
        if(data) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, static_cast<GLsizei>(width), static_cast<GLsizei>(height), 0, GL_RGBA,
                         GL_UNSIGNED_BYTE, data);  // R8G8B8A8
            glGenerateMipmap(GL_TEXTURE_2D);
        }
        glBindTexture(GL_TEXTURE_2D, GL_NONE);
    }
    GLTextureObject(const GLTextureObject&) = delete;
    GLTextureObject(GLTextureObject&&) = delete;
    GLTextureObject& operator=(const GLTextureObject&) = delete;
    GLTextureObject& operator=(GLTextureObject&&) = delete;
    ~GLTextureObject() override {
        glDeleteTextures(1, &mTex);
    }
    [[nodiscard]] TextureId getTexture() const override {
        return mTex;
    }
    [[nodiscard]] Vec2 size() const override {
        return mSize;
    }
};

std::unique_ptr<TextureObject> loadTexture(uint32_t width, uint32_t height, const uint32_t* data) {
    return std::make_unique<GLTextureObject>(width, height, data);
}

class GLCubeMapObject final : public TextureObject {
    GLuint mTex{};
    Vec2 mSize;

public:
    GLCubeMapObject(const uint32_t size, const uint32_t* data) : mSize{ static_cast<float>(size), static_cast<float>(size) } {
        glGenTextures(1, &mTex);
        glBindTexture(GL_TEXTURE_CUBE_MAP, mTex);
        assert(data);
        const auto offset = static_cast<ptrdiff_t>(size) * static_cast<ptrdiff_t>(size);
        for(int32_t idx = 0; idx < 6; ++idx) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + idx, 0, GL_RGBA, static_cast<GLsizei>(size), static_cast<GLsizei>(size),
                         0, GL_RGBA, GL_UNSIGNED_BYTE, data + idx * offset);  // R8G8B8A8
        }
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
        glBindTexture(GL_TEXTURE_CUBE_MAP, GL_NONE);
    }
    GLCubeMapObject(const GLCubeMapObject&) = delete;
    GLCubeMapObject(GLCubeMapObject&&) = delete;
    GLTextureObject& operator=(const GLCubeMapObject&) = delete;
    GLCubeMapObject& operator=(GLCubeMapObject&&) = delete;
    ~GLCubeMapObject() override {
        glDeleteTextures(1, &mTex);
    }
    [[nodiscard]] TextureId getTexture() const override {
        return mTex;
    }
    [[nodiscard]] Vec2 size() const override {
        return mSize;
    }
};

std::unique_ptr<TextureObject> loadCubeMap(uint32_t size, const uint32_t* data) {
    return std::make_unique<GLCubeMapObject>(size, data);
}

class GLVolumeObject final : public TextureObject {
    GLuint mTex{};
    Vec2 mSize;

public:
    GLVolumeObject(uint32_t size, uint32_t channels, const uint8_t* data)
        : mSize{ static_cast<float>(size), static_cast<float>(size) } {
        glGenTextures(1, &mTex);
        glBindTexture(GL_TEXTURE_3D, mTex);
        assert(data);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_BASE_LEVEL, 0);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAX_LEVEL, static_cast<GLint>(std::log2(size)));
        GLenum internalFormat = channels == 1 ? GL_R8 : GL_RGBA;
        GLenum format = channels == 1 ? GL_RED : GL_RGBA;
        glTexImage3D(GL_TEXTURE_3D, 0, internalFormat, static_cast<GLsizei>(size), static_cast<GLsizei>(size),
                     static_cast<GLsizei>(size), 0, format, GL_UNSIGNED_BYTE, data);  // R8G8B8A8
        glGenerateMipmap(GL_TEXTURE_3D);
        glBindTexture(GL_TEXTURE_3D, GL_NONE);
    }
    GLVolumeObject(const GLVolumeObject&) = delete;
    GLVolumeObject(GLVolumeObject&&) = delete;
    GLVolumeObject& operator=(const GLVolumeObject&) = delete;
    GLVolumeObject& operator=(GLVolumeObject&&) = delete;
    ~GLVolumeObject() override {
        glDeleteTextures(1, &mTex);
    }
    [[nodiscard]] TextureId getTexture() const override {
        return mTex;
    }
    [[nodiscard]] Vec2 size() const override {
        return mSize;
    }
};

std::unique_ptr<TextureObject> loadVolume(uint32_t size, uint32_t channels, const uint8_t* data) {
    return std::make_unique<GLVolumeObject>(size, channels, data);
}

struct DynamicTexture final {
    std::unique_ptr<GLTextureObject> tex;
    std::vector<uint32_t> data;
    std::function<void(uint32_t*)> update;
};

class OpenGLPipeline final : public Pipeline {
    GLuint mVAOImage{};
    GLuint mVAOCubeMap{};
    GLuint mVBO{};
    std::vector<std::unique_ptr<FrameBuffer>> mFrameBuffers;
    std::vector<std::unique_ptr<GLCubeMapRenderTarget>> mCubeMapRenderTargets;
    std::vector<std::unique_ptr<RenderPass>> mRenderPasses;
    std::vector<DynamicTexture> mDynamicTextures;
    std::vector<std::unique_ptr<TextureObject>> mTextures;
    AudioInput mAudioInput;
    KeyboardInput mKeyboardInput;

    static uint8_t toByte(const float value) {
        return static_cast<uint8_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
    }

    static float resample(const std::vector<float>& values, const float position) {
        if(values.empty())
            return 0.0f;
        if(values.size() == 1)
            return values.front();
        const float scaled = std::clamp(position, 0.0f, 1.0f) * static_cast<float>(values.size() - 1);
        const auto lo = static_cast<size_t>(scaled);
        const auto hi = std::min(lo + 1, values.size() - 1);
        const float fraction = scaled - static_cast<float>(lo);
        return values[lo] + (values[hi] - values[lo]) * fraction;
    }

    void updateKeyboardTexture(uint32_t* data) const {
        const auto& pixels = mKeyboardInput.pixels();
        std::copy(pixels.begin(), pixels.end(), data);
    }

    void updateAudioTexture(uint32_t* data) const {
        for(uint32_t x = 0; x < AudioInput::TextureWidth; ++x) {
            const float position = static_cast<float>(x) / static_cast<float>(AudioInput::TextureWidth - 1);
            const float spectrum = mAudioInput.available ? resample(mAudioInput.spectrum, position) : 0.0f;
            const float waveform = mAudioInput.available ? resample(mAudioInput.waveform, position) * 0.5f + 0.5f : 0.5f;
            const uint8_t spectrumByte = toByte(spectrum);
            const uint8_t waveformByte = toByte(waveform);
            data[x] = static_cast<uint32_t>(spectrumByte) | (static_cast<uint32_t>(spectrumByte) << 8U) |
                (static_cast<uint32_t>(spectrumByte) << 16U) | 0xff000000U;
            data[AudioInput::TextureWidth + x] = static_cast<uint32_t>(waveformByte) |
                (static_cast<uint32_t>(waveformByte) << 8U) | (static_cast<uint32_t>(waveformByte) << 16U) | 0xff000000U;
        }
    }

public:
    explicit OpenGLPipeline() {
        glGenBuffers(1, &mVBO);
        glBindBuffer(GL_ARRAY_BUFFER, mVBO);

        glGenVertexArrays(1, &mVAOImage);
        glBindVertexArray(mVAOImage);
        glEnableVertexAttribArray(0);
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, pos)));
        glEnableVertexAttribArray(1);
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, coord)));
        glBindVertexArray(0);

        glGenVertexArrays(1, &mVAOCubeMap);
        glBindVertexArray(mVAOCubeMap);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(VertexCubeMap),
                              // NOLINTNEXTLINE(performance-no-int-to-ptr)
                              reinterpret_cast<void*>(offsetof(VertexCubeMap, pos)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(VertexCubeMap),
                              // NOLINTNEXTLINE(performance-no-int-to-ptr)
                              reinterpret_cast<void*>(offsetof(VertexCubeMap, coord)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(VertexCubeMap),
                              // NOLINTNEXTLINE(performance-no-int-to-ptr)
                              reinterpret_cast<void*>(offsetof(VertexCubeMap, point)));
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, GL_NONE);
    }
    OpenGLPipeline(const OpenGLPipeline&) = delete;
    OpenGLPipeline(OpenGLPipeline&&) = delete;
    OpenGLPipeline& operator=(const OpenGLPipeline&) = delete;
    OpenGLPipeline& operator=(OpenGLPipeline&&) = delete;
    ~OpenGLPipeline() override {
        glDeleteVertexArrays(1, &mVAOImage);
        glDeleteVertexArrays(1, &mVAOCubeMap);
        glDeleteBuffers(1, &mVBO);
    }

    FrameBuffer* createFrameBuffer() override {
        mFrameBuffers.push_back(std::make_unique<GLFrameBuffer>());
        return mFrameBuffers.back().get();
    }
    GLCubeMapRenderTarget* createCubeMapRenderTarget() {
        mCubeMapRenderTargets.push_back(std::make_unique<GLCubeMapRenderTarget>());
        return mCubeMapRenderTargets.back().get();
    }
    std::vector<FrameBuffer*> createCubeMapFrameBuffer() override {
        std::vector<FrameBuffer*> buffers;
        const auto target = createCubeMapRenderTarget();

        for(uint32_t idx = 0; idx < 6; ++idx) {
            mFrameBuffers.push_back(std::make_unique<GLCubeMapFrameBuffer>(target->getTexture(), idx));
            buffers.emplace_back(mFrameBuffers.back().get());
        }
        return buffers;
    }

    void addPass(std::string name, const std::string& src, NodeType type, std::vector<DoubleBufferedFB> target,
                 std::vector<Channel> channels, bool clampOutput) override {
        mRenderPasses.push_back(
            std::make_unique<RenderPass>(std::move(name), src, type, std::move(target), std::move(channels), clampOutput));
    }

    void render(const Vec2 frameBufferSize, const Vec2 clipMin, const Vec2 clipMax, Vec2 size,
                const ShaderToyUniform& uniform) override {
        GLint callerFramebuffer = 0;
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &callerFramebuffer);
        for(auto& [tex, data, update] : mDynamicTextures) {
            update(data.data());
            const auto texId = static_cast<GLuint>(tex->getTexture());
            glBindTexture(GL_TEXTURE_2D, texId);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, static_cast<GLsizei>(tex->size().x), static_cast<GLsizei>(tex->size().y), 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, data.data());  // R8G8B8A8
            glBindTexture(GL_TEXTURE_2D, GL_NONE);
        }
        for(const auto& pass : mRenderPasses) {
            if(!pass->hasOffscreenTarget())
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, static_cast<GLuint>(callerFramebuffer));
            pass->render(frameBufferSize, clipMin, clipMax, size, uniform,
                         pass->getType() == NodeType::Image ? mVAOImage : mVAOCubeMap, mVBO);
        }
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, static_cast<GLuint>(callerFramebuffer));
    }

    TextureId createDynamicTexture(uint32_t width, uint32_t height, std::function<void(uint32_t*)> update) {
        mDynamicTextures.push_back(DynamicTexture{
            std::make_unique<GLTextureObject>(width, height, nullptr),
            std::vector<uint32_t>(checkedSizeProduct({ width, height }, "Dynamic texture")),
            std::move(update),
        });
        return mDynamicTextures.back().tex->getTexture();
    }
    TextureId createTexture(const uint32_t width, const uint32_t height, const uint32_t* data) override {
        mTextures.push_back(std::make_unique<GLTextureObject>(width, height, data));
        return mTextures.back()->getTexture();
    }
    TextureId createCubeMap(const uint32_t size, const uint32_t* data) override {
        mTextures.push_back(std::make_unique<GLCubeMapObject>(size, data));
        return mTextures.back()->getTexture();
    }
    TextureId createVolume(const uint32_t size, const uint32_t channels, const uint8_t* data) override {
        mTextures.push_back(std::make_unique<GLVolumeObject>(size, channels, data));
        return mTextures.back()->getTexture();
    }
    TextureId createKeyboardTexture() override {
        return createDynamicTexture(static_cast<uint32_t>(KeyboardInput::KeyCount), static_cast<uint32_t>(KeyboardInput::Rows),
                                    [this](uint32_t* data) { updateKeyboardTexture(data); });
    }
    TextureId createAudioTexture() override {
        return createDynamicTexture(AudioInput::TextureWidth, AudioInput::TextureHeight,
                                    [this](uint32_t* data) { updateAudioTexture(data); });
    }
    void setKeyboardInput(const KeyboardInput& input) override {
        mKeyboardInput = input;
    }
    void setAudioInput(const AudioInput& input) override {
        mAudioInput = input;
    }
    std::vector<uint8_t> renderToBuffer(Vec2 size, const ShaderToyUniform& uniform) override {
        const auto width = checkedPixelDimension(size.x, "Render width");
        const auto height = checkedPixelDimension(size.y, "Render height");

        // Update dynamic textures first, like in the regular render function
        for(auto& [tex, data, update] : mDynamicTextures) {
            update(data.data());
            const auto texId = static_cast<GLuint>(tex->getTexture());
            glBindTexture(GL_TEXTURE_2D, texId);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, static_cast<GLsizei>(tex->size().x), static_cast<GLsizei>(tex->size().y), 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, data.data());  // R8G8B8A8
            glBindTexture(GL_TEXTURE_2D, GL_NONE);
        }

        auto fb = std::make_unique<GLFrameBuffer>();
        fb->bind(width, height);

        // Render all passes. Offscreen passes may bind and unbind their own framebuffer,
        // so explicitly restore the capture framebuffer before any pass that renders
        // directly to the caller target.
        for(const auto& pass : mRenderPasses) {
            if(!pass->hasOffscreenTarget())
                fb->bind(width, height);
            pass->render(size, Vec2{ 0, 0 }, size, size, uniform, pass->getType() == NodeType::Image ? mVAOImage : mVAOCubeMap,
                         mVBO);
        }

        fb->bind(width, height);
        std::vector<uint8_t> buffer(checkedSizeProduct({ width, height, 3U }, "Render readback"));
        glReadPixels(0, 0, static_cast<GLsizei>(width), static_cast<GLsizei>(height), GL_RGB, GL_UNSIGNED_BYTE, buffer.data());
        fb->unbind();
        return buffer;
    }

    std::vector<uint8_t> snapshotPassRgb(const std::string_view passName) override {
        const auto selected = std::find_if(mRenderPasses.begin(), mRenderPasses.end(),
                                           [passName](const auto& pass) { return pass->getName() == passName; });
        if(selected == mRenderPasses.end())
            throw Error("Unknown shader pass: " + std::string(passName));
        return (*selected)->readRgb();
    }

    std::vector<float> snapshotPassRgba32f(const std::string_view passName) override {
        const auto selected = std::find_if(mRenderPasses.begin(), mRenderPasses.end(),
                                           [passName](const auto& pass) { return pass->getName() == passName; });
        if(selected == mRenderPasses.end())
            throw Error("Unknown shader pass: " + std::string(passName));
        return (*selected)->readRgba32f();
    }

    void overridePassRgba8(const std::string_view passName, const uint32_t width, const uint32_t height,
                           const uint8_t* data) override {
        const auto selected = std::find_if(mRenderPasses.begin(), mRenderPasses.end(),
                                           [passName](const auto& pass) { return pass->getName() == passName; });
        if(selected == mRenderPasses.end())
            throw Error("Unknown shader pass: " + std::string(passName));
        (*selected)->overrideRgba8(width, height, data);
    }

    void restorePassRgba32f(const std::string_view passName, const uint32_t width, const uint32_t height,
                            const float* data) override {
        const auto selected = std::find_if(mRenderPasses.begin(), mRenderPasses.end(),
                                           [passName](const auto& pass) { return pass->getName() == passName; });
        if(selected == mRenderPasses.end())
            throw Error("Unknown shader pass: " + std::string(passName));
        (*selected)->restoreRgba32f(width, height, data);
    }
};

std::unique_ptr<Pipeline> createPipeline() {
    // The embedding application owns the OpenGL context, while the renderer
    // owns its loader implementation. A compatible context must be current
    // before a document is compiled.
    if(gladLoadGL() == 0)
        throw Error("Failed to initialize OpenGL loader");
    return std::make_unique<OpenGLPipeline>();
}

SHADERTOY_NAMESPACE_END
