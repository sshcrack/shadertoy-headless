/*
    SPDX-License-Identifier: Apache-2.0
    Copyright 2023-2026 Yingwei Zheng and contributors
*/

#include "shadertoy/ShaderToyContext.hpp"
#include "shadertoy/Error.hpp"

#include <algorithm>
#include <cmath>
#include <ctime>

SHADERTOY_NAMESPACE_BEGIN

ShaderToyContext::ShaderToyContext() {
    resetTime();
}

ShaderToyUniform ShaderToyContext::makeUniform() const {
    return {
        mTime,
        mTimeDelta,
        mFrameRate,
        mFrameCount,
        mMouse,
        mDate,
        { mAudioInput.loudness, mAudioInput.bass, mAudioInput.mid, mAudioInput.treble },
        { mAudioInput.onset, mAudioInput.kick, mAudioInput.snare, mAudioInput.hihat },
        { mAudioInput.bpm, mAudioInput.beatPhase, mAudioInput.beatConfidence, mAudioInput.beatStrength },
        { mAudioInput.stereoWidth, mAudioInput.stereoBalance, mAudioInput.stereoCorrelation, mAudioInput.energyTrend },
        { mAudioInput.drop, mAudioInput.sectionChange, mAudioInput.spectralCentroid, mAudioInput.spectralFlux },
        { mAudioInput.available ? 1.0f : 0.0f, mAudioInput.silence ? 1.0f : 0.0f, mAudioInput.sampleRate, 0.0f },
    };
}

void ShaderToyContext::updateDate() {
    const auto offsetNow = mStartTime +
        std::chrono::duration_cast<SystemClock::duration>(
                               std::chrono::nanoseconds{ static_cast<std::chrono::nanoseconds::rep>(mTime * 1e9) });
    const auto current = SystemClock::to_time_t(offsetNow);
    const auto tm = std::localtime(&current);  // NOLINT(concurrency-mt-unsafe)
    if(!tm)
        return;

    mDate = {
        static_cast<float>(tm->tm_year + 1900),
        static_cast<float>(tm->tm_mon + 1),
        static_cast<float>(tm->tm_mday),
        static_cast<float>(tm->tm_hour * 3600 + tm->tm_min * 60 + tm->tm_sec),
    };
}

void ShaderToyContext::tick(const float frameRate) {
    if(!mRunning)
        return;

    const auto now = SystemClock::now();
    const auto elapsed = static_cast<float>(
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - mStartTime).count()) * 1e-9);
    const auto timeScale = std::exp2(mTimeScale);
    const auto nextTime = elapsed * timeScale;
    mTimeDelta = nextTime - mTime;
    mTime = nextTime;
    ++mFrameCount;
    mFrameRate = frameRate;
    updateDate();
}

void ShaderToyContext::tickFixed(const float deltaSeconds, const float frameRate) {
    if(!mRunning)
        return;

    const float timeScale = std::exp2(mTimeScale);
    mTimeDelta = std::max(0.0f, deltaSeconds) * timeScale;
    mTime += mTimeDelta;
    ++mFrameCount;
    mFrameRate = frameRate;
    updateDate();
}

void ShaderToyContext::pause() {
    if(!mRunning)
        return;
    mRunning = false;
    mTimeDelta = 0.0f;
    mPauseTime = SystemClock::now();
}

void ShaderToyContext::resume() {
    if(mRunning)
        return;
    mRunning = true;
    if(mTime == 0.0f)
        mStartTime = SystemClock::now();
    else
        mStartTime += SystemClock::now() - mPauseTime;
}

void ShaderToyContext::resetTime() {
    mStartTime = SystemClock::now();
    mTime = 0.0f;
    mTimeDelta = 0.0f;
    mTimeScale = 0.0f;
    mFrameCount = 0;
    mFrameRate = 0.0f;
    updateDate();
}

void ShaderToyContext::setPipeline(std::unique_ptr<Pipeline> pipeline) {
    mPipeline = std::move(pipeline);
    if(mPipeline) {
        mPipeline->setAudioInput(mAudioInput);
        mPipeline->setKeyboardInput(mKeyboardInput);
    }
    resetTime();
}

void ShaderToyContext::setMouseInput(const std::optional<MouseInput>& mouse) {
    if(mouse) {
        mMouse.x = mouse->x;
        mMouse.y = mouse->y;
        if(mouse->clicked) {
            mMouse.z = mouse->x;
            mMouse.w = mouse->y;
        } else if(!mouse->down) {
            mMouse.z = -std::fabs(mMouse.z);
            mMouse.w = -std::fabs(mMouse.w);
        }
    } else {
        mMouse.z = -std::fabs(mMouse.z);
        mMouse.w = -std::fabs(mMouse.w);
    }
}

void ShaderToyContext::setKeyboardInput(const KeyboardInput& input) {
    mKeyboardInput = input;
    if(mPipeline)
        mPipeline->setKeyboardInput(mKeyboardInput);
}

void ShaderToyContext::setAudioInput(const AudioInput& input) {
    mAudioInput = input;
    if(mPipeline)
        mPipeline->setAudioInput(mAudioInput);
}

void ShaderToyContext::render(const RenderRegion& region) {
    if(!mPipeline)
        return;
    mPipeline->render(region.framebufferSize, region.clipMin, region.clipMax, region.canvasSize, makeUniform());
}

std::vector<uint8_t> ShaderToyContext::renderToBuffer(const Vec2 size) {
    if(!mPipeline)
        return {};
    return mPipeline->renderToBuffer(size, makeUniform());
}

std::vector<uint8_t> ShaderToyContext::snapshotPassRgb(const std::string_view passName) {
    if(!mPipeline)
        return {};
    return mPipeline->snapshotPassRgb(passName);
}

std::vector<float> ShaderToyContext::snapshotPassRgba32f(const std::string_view passName) {
    if(!mPipeline)
        return {};
    return mPipeline->snapshotPassRgba32f(passName);
}

void ShaderToyContext::reloadPassSource(const std::string_view passName, const std::string& source) {
    if(!mPipeline)
        throw Error("No shader pipeline is loaded");
    mPipeline->reloadPassSource(passName, source);
}

void ShaderToyContext::overridePassRgba8(const std::string_view passName, const uint32_t width, const uint32_t height,
                                         const uint8_t* data) {
    if(!mPipeline)
        return;
    mPipeline->overridePassRgba8(passName, width, height, data);
}

void ShaderToyContext::restorePassRgba32f(const std::string_view passName, const uint32_t width, const uint32_t height,
                                          const float* data) {
    if(!mPipeline)
        return;
    mPipeline->restorePassRgba32f(passName, width, height, data);
}

void ShaderToyContext::setProfilingEnabled(const bool enabled) {
    if(mPipeline)
        mPipeline->setProfilingEnabled(enabled);
}

const std::vector<PassTiming>& ShaderToyContext::lastPassTimings() const {
    static const std::vector<PassTiming> empty;
    return mPipeline ? mPipeline->lastPassTimings() : empty;
}

void ShaderToyContext::setFixedState(const float timeSeconds, const int32_t frame, const float frameRate) {
    mTime = std::max(0.0f, timeSeconds);
    mTimeDelta = 0.0f;
    mFrameCount = std::max(0, frame);
    mFrameRate = std::max(0.0f, frameRate);
    mRunning = true;
    mStartTime = SystemClock::now() -
        std::chrono::duration_cast<SystemClock::duration>(
                     std::chrono::nanoseconds{ static_cast<std::chrono::nanoseconds::rep>(mTime * 1e9) });
    updateDate();
}

SHADERTOY_NAMESPACE_END
