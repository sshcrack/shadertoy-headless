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

#define IMGUI_DEFINE_MATH_OPERATORS
#include "NodeEditor/Builders.hpp"

#include "shadertoy/SuppressWarningPush.hpp"
#include <imgui-node-editor/imgui_node_editor.h>
#include <imgui.h>
#include "shadertoy/SuppressWarningPop.hpp"

namespace ed = ax::NodeEditor;
namespace util = ax::NodeEditor::Utilities;

util::BlueprintNodeBuilder::BlueprintNodeBuilder(const ImTextureID texture, const int textureWidth, const int textureHeight)
    : mHeaderTextureId(texture), mHeaderTextureWidth(textureWidth), mHeaderTextureHeight(textureHeight) {}

void util::BlueprintNodeBuilder::begin(const ed::NodeId id) {
    mCurrentNodeId = id;
    mHeaderMin = {};
    mHeaderMax = {};
    mHeaderOpen = false;
    mPinOpen = false;

    ed::PushStyleVar(ed::StyleVar_NodePadding, ImVec4(8, 4, 8, 8));
    ed::BeginNode(id);
    ImGui::PushID(id.AsPointer());
}

void util::BlueprintNodeBuilder::end() {
    if(mPinOpen)
        endPin();
    if(mHeaderOpen)
        endHeader();

    ed::EndNode();

    if(ImGui::IsItemVisible() && mHeaderMax.x > mHeaderMin.x && mHeaderMax.y > mHeaderMin.y) {
        const auto alpha = static_cast<int>(255.0f * ImGui::GetStyle().Alpha);
        auto* drawList = ed::GetNodeBackgroundDrawList(mCurrentNodeId);
        const auto halfBorderWidth = ed::GetStyle().NodeBorderWidth * 0.5f;
        const auto headerColor = IM_COL32(0, 0, 0, alpha) | (mHeaderColor & IM_COL32(255, 255, 255, 0));

        if(mHeaderTextureId != 0 && mHeaderTextureWidth > 0 && mHeaderTextureHeight > 0) {
            const auto uv = ImVec2((mHeaderMax.x - mHeaderMin.x) / (4.0f * static_cast<float>(mHeaderTextureWidth)),
                                   (mHeaderMax.y - mHeaderMin.y) / (4.0f * static_cast<float>(mHeaderTextureHeight)));
            drawList->AddImageRounded(mHeaderTextureId, mHeaderMin - ImVec2(8 - halfBorderWidth, 4 - halfBorderWidth),
                                      mHeaderMax + ImVec2(8 - halfBorderWidth, 0), ImVec2(0.0f, 0.0f), uv, headerColor,
                                      ed::GetStyle().NodeRounding, ImDrawFlags_RoundCornersTop);
        } else {
            drawList->AddRectFilled(mHeaderMin - ImVec2(8 - halfBorderWidth, 4 - halfBorderWidth),
                                    mHeaderMax + ImVec2(8 - halfBorderWidth, 0), headerColor, ed::GetStyle().NodeRounding,
                                    ImDrawFlags_RoundCornersTop);
        }

        drawList->AddLine(ImVec2(mHeaderMin.x - (8 - halfBorderWidth), mHeaderMax.y - 0.5f),
                          ImVec2(mHeaderMax.x + (8 - halfBorderWidth), mHeaderMax.y - 0.5f),
                          ImColor(255, 255, 255, 96 * alpha / (3 * 255)), 1.0f);
    }

    ImGui::PopID();
    ed::PopStyleVar();
    mCurrentNodeId = 0;
}

void util::BlueprintNodeBuilder::header(const ImVec4& color) {
    mHeaderColor = ImColor(color);
    ImGui::BeginGroup();
    mHeaderOpen = true;
}

void util::BlueprintNodeBuilder::endHeader() {
    if(!mHeaderOpen)
        return;

    ImGui::EndGroup();
    mHeaderMin = ImGui::GetItemRectMin();
    mHeaderMax = ImGui::GetItemRectMax();
    ImGui::Dummy(ImVec2(0.0f, ImGui::GetStyle().ItemSpacing.y));
    mHeaderOpen = false;
}

void util::BlueprintNodeBuilder::pin(const ed::PinId id, const ed::PinKind kind) {
    ed::BeginPin(id, kind);
    ImGui::BeginGroup();
    mPinOpen = true;
}

void util::BlueprintNodeBuilder::endPin() {
    if(!mPinOpen)
        return;
    ImGui::EndGroup();
    ed::EndPin();
    mPinOpen = false;
}

void util::BlueprintNodeBuilder::input(const ed::PinId id) {
    ed::PushStyleVar(ed::StyleVar_PivotAlignment, ImVec2(0.0f, 0.5f));
    ed::PushStyleVar(ed::StyleVar_PivotSize, ImVec2(0.0f, 0.0f));
    pin(id, ed::PinKind::Input);
}

void util::BlueprintNodeBuilder::endInput() {
    endPin();
    ed::PopStyleVar(2);
}

void util::BlueprintNodeBuilder::middle() {
    ImGui::Spacing();
}

void util::BlueprintNodeBuilder::output(const ed::PinId id) {
    ed::PushStyleVar(ed::StyleVar_PivotAlignment, ImVec2(1.0f, 0.5f));
    ed::PushStyleVar(ed::StyleVar_PivotSize, ImVec2(0.0f, 0.0f));
    pin(id, ed::PinKind::Output);
}

void util::BlueprintNodeBuilder::endOutput() {
    endPin();
    ed::PopStyleVar(2);
}
