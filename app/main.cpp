/*
    SPDX-License-Identifier: Apache-2.0
    Copyright 2023-2026 Yingwei Zheng and contributors
    Licensed under the Apache License, Version 2.0 (the "License");
*/

#include "FileDialog.hpp"
#include "NodeEditor/PipelineEditor.hpp"
#include "shadertoy/Config.hpp"
#include "shadertoy/ShaderToy.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "shadertoy/SuppressWarningPush.hpp"

#include <fmt/format.h>
#include <hello_imgui/dpi_aware.h>
#include <hello_imgui/hello_imgui.h>
#include <hello_imgui/hello_imgui_screenshot.h>
#include <httplib.h>
#include <imgui_stdlib.h>
#include <magic_enum/magic_enum.hpp>
#include <nlohmann/json.hpp>
#include <openssl/crypto.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define GL_SILENCE_DEPRECATION
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#ifdef SHADERTOY_WINDOWS
#define NOMINMAX
#include <Windows.h>
#endif

#include "shadertoy/SuppressWarningPop.hpp"

using HelloImGui::EmToVec2;

SHADERTOY_NAMESPACE_BEGIN

namespace {

    struct CanvasDrawState final {
        Runtime* runtime = nullptr;
        Vec2 canvasSize;
        ImVec4 bound{};
    };

    CanvasDrawState canvasDrawState;
    std::optional<std::function<void()>> takeScreenshot;

    [[noreturn]] void fatalUiError(const std::string_view error) {
        fmt::print(stderr, "{}\n", error);
        std::abort();
    }

    bool endsWith(const std::string_view str, const std::string_view pattern) {
        return str.size() >= pattern.size() && str.substr(str.size() - pattern.size()) == pattern;
    }

    bool startsWith(const std::string_view str, const std::string_view pattern) {
        return str.size() >= pattern.size() && str.substr(0, pattern.size()) == pattern;
    }

    void openURL(const std::string& url) {
#if defined(SHADERTOY_WINDOWS)
        ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#elif defined(SHADERTOY_MACOS)
        const auto ret = std::system(("open '" + url + "'").c_str());
        SHADERTOY_UNUSED(ret);
#else
        const auto ret = std::system(("xdg-open '" + url + "'").c_str());
        SHADERTOY_UNUSED(ret);
#endif
    }

    void saveScreenshot(const ImVec4& bound) {
        const auto [width, height, bufferRgb] = HelloImGui::AppWindowScreenshotRgbBuffer();
        if(bufferRgb.empty()) {
            HelloImGui::Log(HelloImGui::LogLevel::Error, "Failed to get screenshot since it is not supported by the backend");
            return;
        }

        const auto beginX = std::max(static_cast<int32_t>(bound.x), 0);
        const auto beginY = std::max(static_cast<int32_t>(bound.y), 0);
        const auto endX = std::min(static_cast<int32_t>(bound.z), static_cast<int32_t>(width));
        const auto endY = std::min(static_cast<int32_t>(bound.w), static_cast<int32_t>(height));
        if(beginX >= endX || beginY >= endY)
            return;

        const auto imageWidth = endX - beginX;
        const auto imageHeight = endY - beginY;
        std::vector<uint8_t> image(static_cast<std::size_t>(imageWidth) * imageHeight * 3U);
        for(auto y = beginY; y < endY; ++y) {
            std::copy(bufferRgb.begin() + static_cast<ptrdiff_t>((static_cast<size_t>(y) * width + beginX) * 3),
                      bufferRgb.begin() + static_cast<ptrdiff_t>((static_cast<size_t>(y) * width + beginX + imageWidth) * 3),
                      image.begin() + static_cast<ptrdiff_t>(static_cast<size_t>(y - beginY) * imageWidth * 3));
        }

        const auto selectedPath = FileDialog::saveFile("Images", "png,jpg,bmp,tga");
        if(!selectedPath)
            return;

        const auto* path = selectedPath->c_str();
        const std::string_view imagePath = *selectedPath;
        const auto* data = image.data();
        const auto stride = imageWidth * 3;
        int result = 0;
        if(endsWith(imagePath, ".png"))
            result = stbi_write_png(path, imageWidth, imageHeight, 3, data, stride);
        else if(endsWith(imagePath, ".jpg"))
            result = stbi_write_jpg(path, imageWidth, imageHeight, 3, data, 90);
        else if(endsWith(imagePath, ".bmp"))
            result = stbi_write_bmp(path, imageWidth, imageHeight, 3, data);
        else if(endsWith(imagePath, ".tga"))
            result = stbi_write_tga(path, imageWidth, imageHeight, 3, data);
        else {
            HelloImGui::Log(HelloImGui::LogLevel::Error, "Unrecognized image format");
            return;
        }

        if(result == 0)
            HelloImGui::Log(HelloImGui::LogLevel::Error, "Failed to save screenshot");
    }

    void renderCanvasCallback(const ImDrawList*, const ImDrawCmd* command) {
        if(!canvasDrawState.runtime)
            return;

        const auto* drawData = ImGui::GetDrawData();
        if(!drawData)
            return;

        const ImVec2 framebufferSize{ drawData->DisplaySize.x * drawData->FramebufferScale.x,
                                      drawData->DisplaySize.y * drawData->FramebufferScale.y };
        const ImVec2 clipOffset = drawData->DisplayPos;
        const ImVec2 clipScale = drawData->FramebufferScale;
        const ImVec2 clipMin((command->ClipRect.x - clipOffset.x) * clipScale.x,
                             (command->ClipRect.y - clipOffset.y) * clipScale.y);
        const ImVec2 clipMax((command->ClipRect.z - clipOffset.x) * clipScale.x,
                             (command->ClipRect.w - clipOffset.y) * clipScale.y);
        if(clipMax.x <= clipMin.x || clipMax.y <= clipMin.y)
            return;

        canvasDrawState.bound = { clipMin.x, clipMin.y, clipMax.x, clipMax.y };
        canvasDrawState.runtime->render(RenderRegion{
            .framebufferSize = { framebufferSize.x, framebufferSize.y },
            .clipMin = { clipMin.x, clipMin.y },
            .clipMax = { clipMax.x, clipMax.y },
            .canvasSize = canvasDrawState.canvasSize,
        });
    }

    void updateKeyboard(Runtime& runtime) {
        static KeyboardInput keyboard;

        struct Mapping final {
            uint8_t virtualKey;
            ImGuiKey imguiKey;
        };

        static constexpr Mapping mapping[] = {
            { 8, ImGuiKey_Backspace },
            { 9, ImGuiKey_Tab },
            { 13, ImGuiKey_Enter },
            { 16, ImGuiKey_LeftShift },
            { 16, ImGuiKey_RightShift },
            { 17, ImGuiKey_LeftCtrl },
            { 17, ImGuiKey_RightCtrl },
            { 19, ImGuiKey_Pause },
            { 20, ImGuiKey_CapsLock },
            { 27, ImGuiKey_Escape },
            { 32, ImGuiKey_Space },
            { 33, ImGuiKey_PageUp },
            { 34, ImGuiKey_PageDown },
            { 35, ImGuiKey_End },
            { 36, ImGuiKey_Home },
            { 37, ImGuiKey_LeftArrow },
            { 38, ImGuiKey_UpArrow },
            { 39, ImGuiKey_RightArrow },
            { 40, ImGuiKey_DownArrow },
            { 44, ImGuiKey_PrintScreen },
            { 45, ImGuiKey_Insert },
            { 46, ImGuiKey_Delete },
            { 48, ImGuiKey_0 },
            { 49, ImGuiKey_1 },
            { 50, ImGuiKey_2 },
            { 51, ImGuiKey_3 },
            { 52, ImGuiKey_4 },
            { 53, ImGuiKey_5 },
            { 54, ImGuiKey_6 },
            { 55, ImGuiKey_7 },
            { 56, ImGuiKey_8 },
            { 57, ImGuiKey_9 },
            { 65, ImGuiKey_A },
            { 66, ImGuiKey_B },
            { 67, ImGuiKey_C },
            { 68, ImGuiKey_D },
            { 69, ImGuiKey_E },
            { 70, ImGuiKey_F },
            { 71, ImGuiKey_G },
            { 72, ImGuiKey_H },
            { 73, ImGuiKey_I },
            { 74, ImGuiKey_J },
            { 75, ImGuiKey_K },
            { 76, ImGuiKey_L },
            { 77, ImGuiKey_M },
            { 78, ImGuiKey_N },
            { 79, ImGuiKey_O },
            { 80, ImGuiKey_P },
            { 81, ImGuiKey_Q },
            { 82, ImGuiKey_R },
            { 83, ImGuiKey_S },
            { 84, ImGuiKey_T },
            { 85, ImGuiKey_U },
            { 86, ImGuiKey_V },
            { 87, ImGuiKey_W },
            { 88, ImGuiKey_X },
            { 89, ImGuiKey_Y },
            { 90, ImGuiKey_Z },
            { 96, ImGuiKey_Keypad0 },
            { 97, ImGuiKey_Keypad1 },
            { 98, ImGuiKey_Keypad2 },
            { 99, ImGuiKey_Keypad3 },
            { 100, ImGuiKey_Keypad4 },
            { 101, ImGuiKey_Keypad5 },
            { 102, ImGuiKey_Keypad6 },
            { 103, ImGuiKey_Keypad7 },
            { 104, ImGuiKey_Keypad8 },
            { 105, ImGuiKey_Keypad9 },
            { 106, ImGuiKey_KeypadMultiply },
            { 107, ImGuiKey_KeypadAdd },
            { 109, ImGuiKey_KeypadSubtract },
            { 110, ImGuiKey_KeypadDecimal },
            { 111, ImGuiKey_KeypadDivide },
            { 112, ImGuiKey_F1 },
            { 113, ImGuiKey_F2 },
            { 114, ImGuiKey_F3 },
            { 115, ImGuiKey_F4 },
            { 116, ImGuiKey_F5 },
            { 117, ImGuiKey_F6 },
            { 118, ImGuiKey_F7 },
            { 119, ImGuiKey_F8 },
            { 120, ImGuiKey_F9 },
            { 121, ImGuiKey_F10 },
            { 122, ImGuiKey_F11 },
            { 123, ImGuiKey_F12 },
            { 144, ImGuiKey_NumLock },
            { 145, ImGuiKey_ScrollLock },
            { 186, ImGuiKey_Semicolon },
            { 187, ImGuiKey_Equal },
            { 188, ImGuiKey_Comma },
            { 189, ImGuiKey_Minus },
            { 190, ImGuiKey_Period },
            { 191, ImGuiKey_Slash },
            { 192, ImGuiKey_GraveAccent },
            { 219, ImGuiKey_LeftBracket },
            { 220, ImGuiKey_Backslash },
            { 221, ImGuiKey_RightBracket },
            { 222, ImGuiKey_Apostrophe },
        };

        std::array<bool, KeyboardInput::KeyCount> down{};
        std::array<bool, KeyboardInput::KeyCount> pressed{};
        for(const auto [virtualKey, imguiKey] : mapping) {
            down[virtualKey] = down[virtualKey] || ImGui::IsKeyDown(imguiKey);
            pressed[virtualKey] = pressed[virtualKey] || ImGui::IsKeyPressed(imguiKey, false);
        }

        keyboard.clearTransient();
        for(std::size_t key = 0; key < KeyboardInput::KeyCount; ++key)
            keyboard.setKey(static_cast<uint8_t>(key), down[key], pressed[key]);
        runtime.setKeyboardInput(keyboard);
    }

    void showCanvas(Runtime& runtime) {
        if(!ImGui::Begin("Canvas", nullptr)) {
            ImGui::End();
            return;
        }

        const auto reservedHeight = ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing();
        ImVec2 size(0, 0);
        if(ImGui::BeginChild("CanvasRegion", ImVec2(0, -reservedHeight), false)) {
            size = ImGui::GetContentRegionAvail();
            const auto base = ImGui::GetCursorScreenPos();

            ImGui::InvisibleButton("CanvasArea", size, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
            std::optional<MouseInput> mouse;
            if(ImGui::IsItemHovered() && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                const auto position = ImGui::GetMousePos();
                mouse = MouseInput{
                    .x = position.x - base.x,
                    .y = size.y - (position.y - base.y),
                    .down = true,
                    .clicked = ImGui::IsMouseClicked(ImGuiMouseButton_Left),
                };
            }
            runtime.setMouseInput(mouse);

            auto* drawList = ImGui::GetWindowDrawList();
            if(runtime.isValid()) {
                canvasDrawState.runtime = &runtime;
                canvasDrawState.canvasSize = { size.x, size.y };
                drawList->AddCallback(renderCanvasCallback, nullptr);
                drawList->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
            } else {
                drawList->AddRect(base, ImVec2{ base.x + size.x, base.y + size.y }, IM_COL32(255, 255, 0, 255));
            }
            ImGui::EndChild();
        }

        ImGui::Separator();
        if(ImGui::Button("Reset"))
            runtime.resetTime();
        ImGui::SameLine();

        if(runtime.isRunning()) {
            if(ImGui::Button("Pause")) {
                HelloImGui::GetRunnerParams()->fpsIdling.enableIdling = true;
                runtime.pause();
            }
        } else if(ImGui::Button("Play")) {
            HelloImGui::GetRunnerParams()->fpsIdling.enableIdling = false;
            runtime.resume();
        }

        const auto mouse = runtime.mouseStatus();
        ImGui::SameLine();
        ImGui::Text("% 6.2f % 9.2f fps % 4d x% 4d [%d %d %d %d]", static_cast<double>(runtime.time()),
                    static_cast<double>(ImGui::GetIO().Framerate), static_cast<int>(size.x), static_cast<int>(size.y),
                    static_cast<int>(mouse.x), static_cast<int>(mouse.y), static_cast<int>(mouse.z), static_cast<int>(mouse.w));

        ImGui::SameLine();
        if(ImGui::Button("Screenshot"))
            takeScreenshot = [] { saveScreenshot(canvasDrawState.bound); };

        ImGui::SameLine();
        ImGui::SetNextItemWidth(100.0f);
        auto timeScale = runtime.timeScale();
        if(ImGui::DragFloat("timescale (log2)", &timeScale, 0.01f, -16.0f, 16.0f, "%.1f"))
            runtime.setTimeScale(timeScale);

        ImGui::End();
    }

    std::string url;
    bool openImportModal = false;
    bool openAboutModal = false;

    void showMenu() {
        if(ImGui::BeginMenu("File")) {
            auto& editor = PipelineEditor::get();
            if(ImGui::MenuItem("New shader"))
                editor.resetPipeline();

            if(ImGui::MenuItem("Open shader")) {
                if(const auto path = FileDialog::openFile("ShaderToy Transmission Format", "sttf"))
                    editor.loadSTTF(*path);
            }

            if(ImGui::MenuItem("Save shader")) {
                if(const auto path = FileDialog::saveFile("ShaderToy Transmission Format", "sttf"))
                    editor.saveSTTF(*path);
            }

            if(ImGui::MenuItem("Import from shadertoy.com"))
                openImportModal = true;

            ImGui::Separator();
            if(ImGui::MenuItem("Exit"))
                HelloImGui::GetRunnerParams()->appShallExit = true;
            ImGui::EndMenu();
        }

        if(ImGui::BeginMenu("Help")) {
            if(ImGui::MenuItem("About"))
                openAboutModal = true;
            ImGui::EndMenu();
        }
    }

    void showImportModal() {
        if(openImportModal) {
            ImGui::OpenPopup("Import Shader");
            if(const auto* text = ImGui::GetClipboardText()) {
                const std::string_view clipboardText = text;
                if(startsWith(clipboardText, "https://www.shadertoy.com/view/"))
                    url = clipboardText;
            }
            openImportModal = false;
        }

        const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

        if(ImGui::BeginPopupModal("Import Shader", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextUnformatted("URL");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImGui::CalcTextSize("https://www.shadertoy.com/view/WWWWWWXXXX").x);
            ImGui::InputText("##Url", &url, ImGuiInputTextFlags_CharsNoBlank);

            if(ImGui::Button("Import", EmToVec2(5, 0))) {
                PipelineEditor::get().loadFromShaderToy(url);
                ImGui::CloseCurrentPopup();
            }
            ImGui::SetItemDefaultFocus();
            ImGui::SameLine();
            if(ImGui::Button("Cancel", EmToVec2(5, 0)))
                ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }
    }

    void showAboutModal() {
        if(openAboutModal) {
            ImGui::OpenPopup("About Shadertoy live viewer");
            openAboutModal = false;
        }

        const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

        if(ImGui::BeginPopupModal("About Shadertoy live viewer", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextUnformatted("Unofficial Shadertoy live viewer " SHADERTOY_VERSION);
            ImGui::Separator();
            ImGui::TextUnformatted("Copyright 2023-2026 Yingwei Zheng and contributors");
            ImGui::TextUnformatted("Licensed under the Apache License, Version 2.0");
            ImGui::TextUnformatted("Build Time: " __DATE__ " " __TIME__);

            if(ImGui::Button(SHADERTOY_URL))
                openURL(SHADERTOY_URL);

            if(ImGui::CollapsingHeader("Config", ImGuiTreeNodeFlags_DefaultOpen)) {
                const auto& io = ImGui::GetIO();
                ImGui::Text("Dear ImGui %s (%d)", IMGUI_VERSION, IMGUI_VERSION_NUM);
                ImGui::Text("Platform: %s", io.BackendPlatformName ? io.BackendPlatformName : "Unknown");
                ImGui::Text("Renderer: %s", io.BackendRendererName ? io.BackendRendererName : "Unknown");
#if HELLOIMGUI_HAS_OPENGL
                ImGui::Text("OpenGL version: %s", glGetString(GL_VERSION));
                ImGui::Text("OpenGL vendor: %s", glGetString(GL_VENDOR));
                ImGui::Text("Graphics device: %s", glGetString(GL_RENDERER));
#endif
                ImGui::TextUnformatted("ImGui Node Editor " IMGUI_NODE_EDITOR_VERSION);
                ImGui::Text("GLFW3 %s", glfwGetVersionString());
                ImGui::Text("fmt %d.%d.%d", FMT_VERSION / 10000, (FMT_VERSION % 10000) / 100, FMT_VERSION % 100);
                ImGui::TextUnformatted("cpp-httplib " CPPHTTPLIB_VERSION);
                ImGui::Text("magic_enum %d.%d.%d", MAGIC_ENUM_VERSION_MAJOR, MAGIC_ENUM_VERSION_MINOR, MAGIC_ENUM_VERSION_PATCH);
                ImGui::Text("nlohmann-json %d.%d.%d", NLOHMANN_JSON_VERSION_MAJOR, NLOHMANN_JSON_VERSION_MINOR,
                            NLOHMANN_JSON_VERSION_PATCH);
                ImGui::TextUnformatted(OpenSSL_version(OPENSSL_VERSION));
            }

            if(ImGui::Button("Close", EmToVec2(5, 0)))
                ImGui::CloseCurrentPopup();
            ImGui::SetItemDefaultFocus();
            ImGui::EndPopup();
        }
    }

}  // namespace

int shaderToyMain(const int argc, char** argv) {
    std::string initialPipeline;
    if(argc == 2)
        initialPipeline = argv[1];

    struct FileDialogGuard final {
        ~FileDialogGuard() {
            FileDialog::shutdown();
        }
    } fileDialogGuard;

    Runtime runtime;
    HelloImGui::RunnerParams runnerParams;
    runnerParams.callbacks.LoadAdditionalFonts = [] {};
    runnerParams.appWindowParams.windowTitle = "ShaderToy live viewer";
    runnerParams.appWindowParams.restorePreviousGeometry = true;
    runnerParams.fpsIdling.enableIdling = false;

    runnerParams.imGuiWindowParams.showStatusBar = true;
    runnerParams.imGuiWindowParams.showStatus_Fps = true;
    runnerParams.callbacks.ShowStatus = [] {};

    runnerParams.imGuiWindowParams.showMenuBar = true;
    runnerParams.imGuiWindowParams.showMenu_App_Quit = false;
    runnerParams.callbacks.ShowMenus = [] { showMenu(); };
    runnerParams.callbacks.ShowGui = [] {
        showImportModal();
        showAboutModal();
    };
    runnerParams.callbacks.PreNewFrame = [] {
        if(takeScreenshot) {
            (*takeScreenshot)();
            takeScreenshot.reset();
        }
    };

    runnerParams.imGuiWindowParams.defaultImGuiWindowType = HelloImGui::DefaultImGuiWindowType::ProvideFullScreenDockSpace;
    runnerParams.imGuiWindowParams.enableViewports = true;

    HelloImGui::DockingSplit splitMainBottom;
    splitMainBottom.initialDock = "MainDockSpace";
    splitMainBottom.newDock = "BottomSpace";
    splitMainBottom.direction = ImGuiDir_Down;
    splitMainBottom.ratio = 0.25f;

    HelloImGui::DockingSplit splitMainLeft;
    splitMainLeft.initialDock = "MainDockSpace";
    splitMainLeft.newDock = "LeftSpace";
    splitMainLeft.direction = ImGuiDir_Left;
    splitMainLeft.ratio = 0.75f;

    runnerParams.dockingParams.dockingSplits = { splitMainBottom, splitMainLeft };

    HelloImGui::DockableWindow canvasWindow;
    canvasWindow.label = "Canvas";
    canvasWindow.dockSpaceName = "LeftSpace";
    canvasWindow.GuiFunction = [&] {
        if(!initialPipeline.empty()) {
            if(startsWith(initialPipeline, "https://"))
                PipelineEditor::get().loadFromShaderToy(initialPipeline);
            else if(endsWith(initialPipeline, ".sttf"))
                PipelineEditor::get().loadSTTF(initialPipeline);
            else
                HelloImGui::Log(HelloImGui::LogLevel::Error, "Unrecognized filepath %s", initialPipeline.c_str());
            initialPipeline.clear();
        }

        updateKeyboard(runtime);
        runtime.tick(ImGui::GetIO().Framerate);
        showCanvas(runtime);
    };

    HelloImGui::DockableWindow outputWindow;
    outputWindow.label = "Output";
    outputWindow.dockSpaceName = "BottomSpace";
    outputWindow.GuiFunction = [] { HelloImGui::LogGui(); };

    HelloImGui::DockableWindow editorWindow;
    editorWindow.label = "Editor";
    editorWindow.dockSpaceName = "MainDockSpace";
    editorWindow.GuiFunction = [&] { PipelineEditor::get().render(runtime); };
    runnerParams.dockingParams.dockableWindows = { canvasWindow, outputWindow, editorWindow };

    runnerParams.callbacks.PostInit = [] {
        glewExperimental = GL_TRUE;
        if(glewInit() != GLEW_OK)
            fatalUiError("Failed to initialize GLEW");
        glGetError();
    };

    HelloImGui::Run(runnerParams);
    return 0;
}

SHADERTOY_NAMESPACE_END

int main(const int argc, char** argv) {
    return ShaderToy::shaderToyMain(argc, argv);
}
