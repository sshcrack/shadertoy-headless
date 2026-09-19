//------------------------------------------------------------------------------
// LICENSE
//   This software is dual-licensed to the public domain and under the following
//   license: you are granted a perpetual, irrevocable license to copy, modify,
//   publish, and distribute this file as you see fit.
//
// CREDITS
//   Written by Michal Cichon
//   Modified by Yingwei Zheng
//------------------------------------------------------------------------------

#pragma once

#include "shadertoy/SuppressWarningPush.hpp"
#include <imgui-node-editor/imgui_node_editor.h>
#include "shadertoy/SuppressWarningPop.hpp"

namespace ax::NodeEditor::Utilities {

    /// Small adapter around imgui-node-editor that keeps the original blueprint
    /// presentation without depending on the historical ImGui layout fork.
    struct BlueprintNodeBuilder {
        explicit BlueprintNodeBuilder(ImTextureID texture = 0, int textureWidth = 0, int textureHeight = 0);

        void begin(NodeId id);
        void end();

        void header(const ImVec4& color = ImVec4(1, 1, 1, 1));
        void endHeader();

        void input(PinId id);
        void endInput();

        void middle();

        void output(PinId id);
        void endOutput();

    private:
        void pin(PinId id, ax::NodeEditor::PinKind kind);
        void endPin();

        ImTextureID mHeaderTextureId{};
        int mHeaderTextureWidth{};
        int mHeaderTextureHeight{};
        NodeId mCurrentNodeId{};
        ImU32 mHeaderColor{};
        ImVec2 mHeaderMin{};
        ImVec2 mHeaderMax{};
        bool mHeaderOpen = false;
        bool mPinOpen = false;
    };

}  // namespace ax::NodeEditor::Utilities
