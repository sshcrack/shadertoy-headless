/*
    SPDX-License-Identifier: Apache-2.0
    Copyright 2023-2026 Yingwei Zheng and contributors
*/
#include <shadertoy/shadertoy.h>

#include <GLFW/glfw3.h>

#include <shadertoy/Project.hpp>
#include <shadertoy/ShaderToy.hpp>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    thread_local std::string lastError;

    void setError(const std::string& message) {
        lastError = message;
    }

    template <typename F>
    int guard(F&& fn) {
        try {
            fn();
            lastError.clear();
            return 0;
        } catch(const std::exception& error) {
            setError(error.what());
            return -1;
        } catch(...) {
            setError("Unknown native error");
            return -1;
        }
    }

    std::mutex glfwMutex;
    std::size_t glfwUsers = 0;

    ShaderToy::ProjectPassKind passKind(const st_pass_kind kind) {
        switch(kind) {
            case ST_PASS_IMAGE:
                return ShaderToy::ProjectPassKind::Image;
            case ST_PASS_BUFFER:
                return ShaderToy::ProjectPassKind::Buffer;
            case ST_PASS_CUBEMAP:
                return ShaderToy::ProjectPassKind::CubeMap;
        }
        throw std::runtime_error("Unknown pass kind");
    }

    ShaderToy::ProjectInputKind inputKind(const st_input_kind kind) {
        switch(kind) {
            case ST_INPUT_PASS:
                return ShaderToy::ProjectInputKind::Pass;
            case ST_INPUT_TEXTURE:
                return ShaderToy::ProjectInputKind::Texture;
            case ST_INPUT_KEYBOARD:
                return ShaderToy::ProjectInputKind::Keyboard;
            case ST_INPUT_MUSIC:
                return ShaderToy::ProjectInputKind::Music;
            case ST_INPUT_CUBEMAP:
                return ShaderToy::ProjectInputKind::CubeMap;
            case ST_INPUT_VOLUME:
                return ShaderToy::ProjectInputKind::Volume;
        }
        throw std::runtime_error("Unknown input kind");
    }

    ShaderToy::Filter filterKind(const st_filter filter) {
        switch(filter) {
            case ST_FILTER_MIPMAP:
                return ShaderToy::Filter::Mipmap;
            case ST_FILTER_LINEAR:
                return ShaderToy::Filter::Linear;
            case ST_FILTER_NEAREST:
                return ShaderToy::Filter::Nearest;
        }
        throw std::runtime_error("Unknown filter kind");
    }

    ShaderToy::Wrap wrapKind(const st_wrap wrap) {
        switch(wrap) {
            case ST_WRAP_CLAMP:
                return ShaderToy::Wrap::Clamp;
            case ST_WRAP_REPEAT:
                return ShaderToy::Wrap::Repeat;
        }
        throw std::runtime_error("Unknown wrap kind");
    }

    void copyRgb(const std::vector<uint8_t>& pixels, uint8_t* out, const size_t outLen) {
        if(!out)
            throw std::runtime_error("Output buffer is null");
        if(outLen != pixels.size())
            throw std::runtime_error("Output buffer has the wrong size");
        std::copy(pixels.begin(), pixels.end(), out);
    }

    size_t checkedImageValueCount(const uint32_t width, const uint32_t height, const size_t channels, const char* label) {
        if(width == 0 || height == 0)
            throw std::runtime_error(std::string(label) + " dimensions must be positive");
        if(static_cast<size_t>(width) > std::numeric_limits<size_t>::max() / static_cast<size_t>(height))
            throw std::runtime_error(std::string(label) + " dimensions overflow addressable memory");
        const auto pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
        if(channels == 0 || pixels > std::numeric_limits<size_t>::max() / channels)
            throw std::runtime_error(std::string(label) + " payload size overflows addressable memory");
        return pixels * channels;
    }
}  // namespace

struct st_context {
    GLFWwindow* window{};
};

struct st_project {
    ShaderToy::ProjectDescription description;
};

struct st_runtime {
    ShaderToy::Runtime runtime;
    ShaderToy::KeyboardInput keyboard;
};

extern "C" {

const char* st_last_error(void) {
    return lastError.c_str();
}

st_context* st_context_create_hidden(const uint32_t width, const uint32_t height) {
    try {
        if(width == 0 || height == 0)
            throw std::runtime_error("Context dimensions must be positive");
        if(width > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
           height > static_cast<uint32_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("Context dimensions exceed GLFW's signed integer range");

        {
            std::scoped_lock lock(glfwMutex);
            if(glfwUsers == 0) {
                glfwSetErrorCallback([](const int, const char* message) noexcept {
                    if(message)
                        setError(std::string("GLFW: ") + message);
                });
                if(!glfwInit())
                    throw std::runtime_error(lastError.empty() ? "Failed to initialize GLFW" : lastError);
            }
            ++glfwUsers;
        }

        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif
        auto* window = glfwCreateWindow(static_cast<int>(width), static_cast<int>(height), "shadertoy-cli", nullptr, nullptr);
        if(!window) {
            std::scoped_lock lock(glfwMutex);
            if(--glfwUsers == 0)
                glfwTerminate();
            throw std::runtime_error(lastError.empty() ? "Failed to create hidden OpenGL context" : lastError);
        }
        glfwMakeContextCurrent(window);
        lastError.clear();
        return new st_context{ window };
    } catch(const std::exception& error) {
        setError(error.what());
        return nullptr;
    } catch(...) {
        setError("Unknown context creation error");
        return nullptr;
    }
}

int st_context_make_current(st_context* context) {
    return guard([&] {
        if(!context || !context->window)
            throw std::runtime_error("Context is null");
        glfwMakeContextCurrent(context->window);
    });
}

void st_context_destroy(st_context* context) {
    if(!context)
        return;
    if(context->window)
        glfwDestroyWindow(context->window);
    delete context;

    std::scoped_lock lock(glfwMutex);
    if(glfwUsers > 0 && --glfwUsers == 0)
        glfwTerminate();
}

st_project* st_project_create(const char* name) {
    try {
        if(!name || !*name)
            throw std::runtime_error("Project name must not be empty");
        auto project = std::make_unique<st_project>();
        project->description.name = name;
        lastError.clear();
        return project.release();
    } catch(const std::exception& error) {
        setError(error.what());
        return nullptr;
    }
}

void st_project_destroy(st_project* project) {
    delete project;
}

int st_project_add_pass(st_project* project, const char* name, const st_pass_kind kind, const char* source) {
    return guard([&] {
        if(!project)
            throw std::runtime_error("Project is null");
        if(!name || !*name)
            throw std::runtime_error("Pass name must not be empty");
        if(!source || !*source)
            throw std::runtime_error("Pass source must not be empty");
        project->description.passes.push_back(ShaderToy::ProjectPass{ name, passKind(kind), source, {} });
    });
}

int st_project_add_input(st_project* project, const char* passName, const uint32_t channel, const st_input_kind kind,
                         const char* source, const int previousFrame, const st_filter filter, const st_wrap wrap) {
    return guard([&] {
        if(!project)
            throw std::runtime_error("Project is null");
        if(!passName || !*passName)
            throw std::runtime_error("Pass name must not be empty");
        const auto pass = std::find_if(project->description.passes.begin(), project->description.passes.end(),
                                       [&](const auto& value) { return value.name == passName; });
        if(pass == project->description.passes.end())
            throw std::runtime_error("Unknown pass: " + std::string(passName));

        std::string sourceName;
        if(kind == ST_INPUT_PASS || kind == ST_INPUT_TEXTURE || kind == ST_INPUT_CUBEMAP || kind == ST_INPUT_VOLUME) {
            if(!source || !*source)
                throw std::runtime_error("Pass/resource input source must not be empty");
            sourceName = source;
        }

        pass->inputs.push_back(ShaderToy::ProjectInput{
            channel,
            inputKind(kind),
            std::move(sourceName),
            previousFrame != 0,
            filterKind(filter),
            wrapKind(wrap),
        });
    });
}

int st_project_add_texture_rgba8(st_project* project, const char* name, const uint32_t width, const uint32_t height,
                                 const uint8_t* rgba, const size_t rgbaLen) {
    return guard([&] {
        if(!project)
            throw std::runtime_error("Project is null");
        if(!name || !*name)
            throw std::runtime_error("Texture name must not be empty");
        if(!rgba)
            throw std::runtime_error("Texture data is null");
        const auto expected = checkedImageValueCount(width, height, 4U, "Texture");
        const auto pixelCount = expected / 4U;
        if(rgbaLen != expected)
            throw std::runtime_error("Texture RGBA8 payload has the wrong size");

        std::vector<uint32_t> pixels(pixelCount);
        for(size_t index = 0; index < pixelCount; ++index) {
            const auto offset = index * 4U;
            pixels[index] = static_cast<uint32_t>(rgba[offset]) | (static_cast<uint32_t>(rgba[offset + 1]) << 8U) |
                (static_cast<uint32_t>(rgba[offset + 2]) << 16U) | (static_cast<uint32_t>(rgba[offset + 3]) << 24U);
        }
        project->description.textures.push_back(ShaderToy::ProjectTexture{ name, width, height, std::move(pixels) });
    });
}

int st_project_add_cubemap_rgba8(st_project* project, const char* name, const uint32_t size, const uint8_t* rgba,
                                 const size_t rgbaLen) {
    return guard([&] {
        if(!project)
            throw std::runtime_error("Project is null");
        if(!name || !*name)
            throw std::runtime_error("Cubemap name must not be empty");
        if(!rgba)
            throw std::runtime_error("Cubemap data is null");
        const auto faceBytes = checkedImageValueCount(size, size, 4U, "Cubemap");
        if(faceBytes > std::numeric_limits<size_t>::max() / 6U)
            throw std::runtime_error("Cubemap payload size overflows addressable memory");
        const auto expected = faceBytes * 6U;
        if(rgbaLen != expected)
            throw std::runtime_error("Cubemap RGBA8 payload has the wrong size");

        const auto pixelCount = expected / 4U;
        std::vector<uint32_t> pixels(pixelCount);
        for(size_t index = 0; index < pixelCount; ++index) {
            const auto offset = index * 4U;
            pixels[index] = static_cast<uint32_t>(rgba[offset]) | (static_cast<uint32_t>(rgba[offset + 1]) << 8U) |
                (static_cast<uint32_t>(rgba[offset + 2]) << 16U) | (static_cast<uint32_t>(rgba[offset + 3]) << 24U);
        }
        project->description.cubeMaps.push_back(ShaderToy::ProjectCubeMap{ name, size, std::move(pixels) });
    });
}

int st_project_add_volume_u8(st_project* project, const char* name, const uint32_t size, const uint32_t channels,
                             const uint8_t* data, const size_t dataLen) {
    return guard([&] {
        if(!project)
            throw std::runtime_error("Project is null");
        if(!name || !*name)
            throw std::runtime_error("Volume name must not be empty");
        if(!data)
            throw std::runtime_error("Volume data is null");
        if(channels != 1U && channels != 4U)
            throw std::runtime_error("Volume channels must be 1 or 4");
        const auto plane = checkedImageValueCount(size, size, channels, "Volume");
        if(static_cast<size_t>(size) > std::numeric_limits<size_t>::max() / plane)
            throw std::runtime_error("Volume payload size overflows addressable memory");
        const auto expected = plane * static_cast<size_t>(size);
        if(dataLen != expected)
            throw std::runtime_error("Volume payload has the wrong size");
        project->description.volumes.push_back(
            ShaderToy::ProjectVolume{ name, size, channels, std::vector<uint8_t>(data, data + dataLen) });
    });
}

st_runtime* st_runtime_create(void) {
    try {
        auto runtime = std::make_unique<st_runtime>();
        lastError.clear();
        return runtime.release();
    } catch(const std::exception& error) {
        setError(error.what());
        return nullptr;
    }
}

void st_runtime_destroy(st_runtime* runtime) {
    delete runtime;
}

int st_runtime_load_project(st_runtime* runtime, const st_project* project) {
    return guard([&] {
        if(!runtime || !project)
            throw std::runtime_error("Runtime/project is null");
        auto document = ShaderToy::makeProjectDocument(project->description);
        if(!document)
            throw document.error();
        auto result = runtime->runtime.setDocument(std::move(*document));
        if(!result)
            throw result.error();
    });
}

int st_runtime_save_sttf(const st_runtime* runtime, const char* path) {
    return guard([&] {
        if(!runtime)
            throw std::runtime_error("Runtime is null");
        if(!path || !*path)
            throw std::runtime_error("Output path must not be empty");
        auto result = runtime->runtime.saveSTTF(path);
        if(!result)
            throw result.error();
    });
}

void st_runtime_tick(st_runtime* runtime, const float frameRate) {
    if(runtime)
        runtime->runtime.tick(frameRate);
}

void st_runtime_tick_fixed(st_runtime* runtime, const float deltaSeconds, const float frameRate) {
    if(runtime)
        runtime->runtime.tickFixed(deltaSeconds, frameRate);
}

void st_runtime_reset_time(st_runtime* runtime) {
    if(runtime)
        runtime->runtime.resetTime();
}

void st_runtime_set_fixed_state(st_runtime* runtime, const float timeSeconds, const int32_t frame, const float frameRate) {
    if(runtime)
        runtime->runtime.setFixedState(timeSeconds, frame, frameRate);
}

float st_runtime_time(const st_runtime* runtime) {
    return runtime ? runtime->runtime.time() : 0.0f;
}

int32_t st_runtime_frame(const st_runtime* runtime) {
    return runtime ? runtime->runtime.frame() : 0;
}

void st_runtime_pause(st_runtime* runtime) {
    if(runtime)
        runtime->runtime.pause();
}

void st_runtime_resume(st_runtime* runtime) {
    if(runtime)
        runtime->runtime.resume();
}

int st_runtime_is_running(const st_runtime* runtime) {
    return runtime && runtime->runtime.isRunning() ? 1 : 0;
}

float st_runtime_time_scale(const st_runtime* runtime) {
    return runtime ? runtime->runtime.timeScale() : 0.0f;
}

void st_runtime_set_time_scale(st_runtime* runtime, const float log2Scale) {
    if(runtime)
        runtime->runtime.setTimeScale(log2Scale);
}

int st_runtime_set_mouse(st_runtime* runtime, const float x, const float y, const int down, const int clicked) {
    return guard([&] {
        if(!runtime)
            throw std::runtime_error("Runtime is null");
        runtime->runtime.setMouseInput(ShaderToy::MouseInput{ x, y, down != 0, clicked != 0 });
    });
}

int st_runtime_clear_mouse(st_runtime* runtime) {
    return guard([&] {
        if(!runtime)
            throw std::runtime_error("Runtime is null");
        runtime->runtime.setMouseInput(std::nullopt);
    });
}

int st_runtime_set_key(st_runtime* runtime, const uint8_t key, const int down, const int pressed) {
    return guard([&] {
        if(!runtime)
            throw std::runtime_error("Runtime is null");
        runtime->keyboard.setKey(key, down != 0, pressed != 0);
        runtime->runtime.setKeyboardInput(runtime->keyboard);
    });
}

int st_runtime_clear_key_transients(st_runtime* runtime) {
    return guard([&] {
        if(!runtime)
            throw std::runtime_error("Runtime is null");
        runtime->keyboard.clearTransient();
        runtime->runtime.setKeyboardInput(runtime->keyboard);
    });
}

int st_runtime_render_rgb(st_runtime* runtime, const uint32_t width, const uint32_t height, uint8_t* outRgb,
                          const size_t outLen) {
    return guard([&] {
        if(!runtime)
            throw std::runtime_error("Runtime is null");
        if(!outRgb)
            throw std::runtime_error("Output buffer is null");
        const auto expected = checkedImageValueCount(width, height, 3U, "Render");
        if(outLen != expected)
            throw std::runtime_error("Output buffer has the wrong size");
        const auto pixels =
            runtime->runtime.renderToBuffer(ShaderToy::Vec2{ static_cast<float>(width), static_cast<float>(height) });
        copyRgb(pixels, outRgb, outLen);
    });
}

int st_runtime_snapshot_pass_rgb(st_runtime* runtime, const char* passName, uint8_t* outRgb, const size_t outLen) {
    return guard([&] {
        if(!runtime)
            throw std::runtime_error("Runtime is null");
        if(!passName || !*passName)
            throw std::runtime_error("Pass name must not be empty");
        if(!outRgb)
            throw std::runtime_error("Output buffer is null");
        auto result = runtime->runtime.snapshotPassRgb(passName);
        if(!result)
            throw result.error();
        copyRgb(*result, outRgb, outLen);
    });
}

int st_runtime_snapshot_pass_rgba32f(st_runtime* runtime, const char* passName, float* outRgba, const size_t outLen) {
    return guard([&] {
        if(!runtime)
            throw std::runtime_error("Runtime is null");
        if(!passName || !*passName)
            throw std::runtime_error("Pass name must not be empty");
        if(!outRgba)
            throw std::runtime_error("Output buffer is null");
        auto result = runtime->runtime.snapshotPassRgba32f(passName);
        if(!result)
            throw result.error();
        if(result->size() != outLen)
            throw std::runtime_error("Output buffer has the wrong size");
        std::copy(result->begin(), result->end(), outRgba);
    });
}

int st_runtime_override_pass_rgba8(st_runtime* runtime, const char* passName, const uint32_t width, const uint32_t height,
                                   const uint8_t* rgba, const size_t rgbaLen) {
    return guard([&] {
        if(!runtime)
            throw std::runtime_error("Runtime is null");
        if(!passName || !*passName)
            throw std::runtime_error("Pass name must not be empty");
        if(!rgba)
            throw std::runtime_error("Pass override data is null");
        if(rgbaLen != checkedImageValueCount(width, height, 4U, "Pass override"))
            throw std::runtime_error("Pass override RGBA8 payload has the wrong size");
        std::vector<uint8_t> pixels(rgba, rgba + rgbaLen);
        auto result = runtime->runtime.overridePassRgba8(passName, width, height, pixels);
        if(!result)
            throw result.error();
    });
}

int st_runtime_restore_pass_rgba32f(st_runtime* runtime, const char* passName, const uint32_t width, const uint32_t height,
                                    const float* rgba, const size_t rgbaLen) {
    return guard([&] {
        if(!runtime)
            throw std::runtime_error("Runtime is null");
        if(!passName || !*passName)
            throw std::runtime_error("Pass name must not be empty");
        if(!rgba)
            throw std::runtime_error("Pass state data is null");
        if(rgbaLen != checkedImageValueCount(width, height, 4U, "Pass state"))
            throw std::runtime_error("Pass state RGBA32F payload has the wrong size");
        std::vector<float> pixels(rgba, rgba + rgbaLen);
        auto result = runtime->runtime.restorePassRgba32f(passName, width, height, pixels);
        if(!result)
            throw result.error();
    });
}

}  // extern "C"
