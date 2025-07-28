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

#include "shadertoy/Config.hpp"
#include "shadertoy/NodeEditor/PipelineEditor.hpp"
#include "shadertoy/ShaderToyContext.hpp"
#include "shadertoy/Support.hpp"
#include <cstdlib>

#include "shadertoy/SuppressWarningPush.hpp"

#include <fmt/format.h>
#include <hello_imgui/dpi_aware.h>
#include <hello_imgui/hello_imgui.h>
#include <hello_imgui/hello_imgui_screenshot.h>
#include <httplib.h>
#include <magic_enum/magic_enum.hpp>
#include <nlohmann/json.hpp>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <iostream>

#define GL_SILENCE_DEPRECATION  // NOLINT(clang-diagnostic-unused-macros)
#include <GL/glew.h>

#include <GLFW/glfw3.h>
#ifdef SHADERTOY_WINDOWS
#define NOMINMAX  // NOLINT(clang-diagnostic-unused-macros)
#include <Windows.h>
#endif

using HelloImGui::EmToVec2;

#include <string_view>
#include <optional>

#include "shadertoy/SuppressWarningPop.hpp"
SHADERTOY_NAMESPACE_BEGIN

static bool startsWith(const std::string_view& str, const std::string_view& pattern) {
    return str.size() >= pattern.size() && str.substr(0, pattern.size()) == pattern;
}

static void showCanvas(ShaderToyContext& ctx) {
    if(!ImGui::Begin("Canvas", nullptr)) {
        ImGui::End();
        return;
    }

    const auto reservedHeight = ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing();
    ImVec2 size(0, 0);
    if(ImGui::BeginChild("CanvasRegion", ImVec2(0, -reservedHeight), false)) {
        size = ImGui::GetContentRegionAvail();
        const auto base = ImGui::GetCursorScreenPos();
        std::optional<ImVec4> mouse = std::nullopt;
        ImGui::InvisibleButton("CanvasArea", size, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);

        ctx.render(base, size, mouse);
        ImGui::EndChild();
    }

    ImGui::End();
}

int shaderToyMain(int argc, char** argv) {
    std::string initialPipeline = "https://www.shadertoy.com/view/wlG3RW";//"https://www.shadertoy.com/view/Xlcczj";//"https://www.shadertoy.com/view/WsSBzh";

    if(argc == 2) {
        initialPipeline = argv[1];
    }

    ShaderToyContext ctx;
    HelloImGui::RunnerParams runnerParams;
    runnerParams.appWindowParams.windowTitle = "ShaderToy live viewer";
    runnerParams.appWindowParams.restorePreviousGeometry = true;
    runnerParams.fpsIdling.enableIdling = false;

    runnerParams.imGuiWindowParams.showStatusBar = true;
    runnerParams.imGuiWindowParams.showStatus_Fps = true;
    runnerParams.callbacks.ShowStatus = [&] {};

    runnerParams.imGuiWindowParams.showMenuBar = true;
    runnerParams.imGuiWindowParams.showMenu_App_Quit = false;

    runnerParams.callbacks.LoadAdditionalFonts = [] { HelloImGui::ImGuiDefaultSettings::LoadDefaultFont_WithFontAwesomeIcons(); };

    runnerParams.callbacks.ShowGui = [&] {
        if(!initialPipeline.empty()) {
            if(startsWith(initialPipeline, "https://")) {
                auto res = PipelineEditor::get().loadFromShaderToy(initialPipeline);
                if(!res) {
                    HelloImGui::Log(HelloImGui::LogLevel::Error, "Failed to load pipeline from ShaderToy: %s", res.error().what());
                } else {
                    HelloImGui::Log(HelloImGui::LogLevel::Info, "Loaded pipeline from ShaderToy: %s", initialPipeline.c_str());
                }
            } else {
                HelloImGui::Log(HelloImGui::LogLevel::Error, "Unrecognized filepath %s", initialPipeline.c_str());
            }

            initialPipeline.clear();
        }

        if(auto res = PipelineEditor::get().update(ctx); !res) {
            HelloImGui::Log(HelloImGui::LogLevel::Error, "Could not update Pipeline: %s", res.error().what());
            std::cerr << res.error().what() << std::endl;
        }
        ctx.tick();
        showCanvas(ctx);

        ImGui::Begin("Console");
        HelloImGui::LogGui();
        ImGui::End();
    };

    // 8x MSAA
    runnerParams.callbacks.PostInit = [] {
        if(glewInit() != GLEW_OK)
            reportFatalError("Failed to initialize glew");
    };
    HelloImGui::Run(runnerParams);
    return 0;
}

SHADERTOY_NAMESPACE_END

int main(const int argc, char** argv) {
    return ShaderToy::shaderToyMain(argc, argv);
}
