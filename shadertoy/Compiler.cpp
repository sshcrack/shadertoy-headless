/*
    SPDX-License-Identifier: Apache-2.0
    Copyright 2023-2026 Yingwei Zheng and contributors
*/

#include "shadertoy/Compiler.hpp"
#include "shadertoy/Support.hpp"

#include <algorithm>
#include <cassert>
#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

SHADERTOY_NAMESPACE_BEGIN

std::unique_ptr<Pipeline> compilePipeline(const ShaderDocument& document) {
    using Inputs = std::vector<const Link*>;
    std::unordered_map<const Node*, Inputs> inputs;
    std::unordered_set<const Node*> allNodes;
    for(const auto& node : document.nodes)
        allNodes.insert(node.get());

    const Node* sinkNode = nullptr;
    const Node* directRenderNode = nullptr;
    std::size_t outputLinks = 0;

    for(const auto& link : document.links) {
        if(!link.start || !link.end || !allNodes.contains(link.start) || !allNodes.contains(link.end))
            throw Error("Pipeline contains a link with an invalid endpoint");
        inputs[link.end].push_back(&link);
        if(link.end->getNodeClass() == NodeClass::RenderOutput) {
            ++outputLinks;
            sinkNode = link.end;
            directRenderNode = link.start;
        }
    }

    if(outputLinks != 1 || !sinkNode || !directRenderNode || directRenderNode->getNodeClass() != NodeClass::GLSLShader) {
        throw Error("Exactly one shader must be connected to the final render output");
    }

    enum class VisitState : uint8_t { Visiting, Done };
    std::unordered_map<const Node*, VisitState> state;
    std::vector<const Node*> order;
    std::unordered_set<const Node*> requiredDoubleBuffer;
    std::vector<const LastFrame*> reachableLastFrames;

    std::function<void(const Node*)> visit = [&](const Node* node) {
        if(const auto found = state.find(node); found != state.end()) {
            if(found->second == VisitState::Visiting)
                throw Error("Loop detected in current-frame shader dependencies");
            return;
        }

        state.emplace(node, VisitState::Visiting);
        if(const auto found = inputs.find(node); found != inputs.end()) {
            for(const auto* link : found->second)
                visit(link->start);
        }

        if(node->getNodeClass() == NodeClass::LastFrame) {
            const auto& lastFrame = dynamic_cast<const LastFrame&>(*node);
            if(!lastFrame.refNode || !allNodes.contains(lastFrame.refNode))
                throw Error("LastFrame references an invalid node");
            if(lastFrame.refNode == directRenderNode)
                throw Error("The final image pass cannot be used as a LastFrame source");
            requiredDoubleBuffer.insert(lastFrame.refNode);
            reachableLastFrames.push_back(&lastFrame);
        }

        state[node] = VisitState::Done;
        order.push_back(node);
    };

    visit(sinkNode);

    // A LastFrame edge is deliberately not a current-frame dependency. Its
    // referenced pass still needs to execute, however, even if it is otherwise
    // only reachable through the feedback edge.
    for(std::size_t i = 0; i < reachableLastFrames.size(); ++i)
        visit(reachableLastFrames[i]->refNode);

    auto pipeline = createPipeline();
    if(!pipeline)
        throw Error("Failed to create the OpenGL pipeline");

    std::unordered_map<const Node*, DoubleBufferedTex> textureMap;
    std::unordered_map<const Node*, Vec2> textureSizeMap;
    std::unordered_map<const Node*, std::vector<DoubleBufferedFB>> frameBufferMap;

    // Allocate render targets first so LastFrame nodes can refer to a pass that
    // appears later in execution order.
    for(const auto* node : order) {
        if(node->getNodeClass() != NodeClass::GLSLShader)
            continue;

        if(node->getNodeType() == NodeType::Image) {
            DoubleBufferedFB target{ nullptr };
            if(requiredDoubleBuffer.contains(node)) {
                target = DoubleBufferedFB{ pipeline->createFrameBuffer(), pipeline->createFrameBuffer() };
            } else if(node != directRenderNode) {
                target = DoubleBufferedFB{ pipeline->createFrameBuffer() };
            }
            frameBufferMap.emplace(node, std::vector<DoubleBufferedFB>{ target });
        } else if(node->getNodeType() == NodeType::CubeMap) {
            if(node == directRenderNode)
                throw Error("A cubemap pass cannot be the final image output");

            std::vector<DoubleBufferedFB> targets;
            targets.reserve(6);
            if(requiredDoubleBuffer.contains(node)) {
                const auto first = pipeline->createCubeMapFrameBuffer();
                const auto second = pipeline->createCubeMapFrameBuffer();
                if(first.size() != 6 || second.size() != 6)
                    throw Error("Invalid cubemap framebuffer allocation");
                for(uint32_t face = 0; face < 6; ++face)
                    targets.emplace_back(first[face], second[face]);
            } else {
                const auto first = pipeline->createCubeMapFrameBuffer();
                if(first.size() != 6)
                    throw Error("Invalid cubemap framebuffer allocation");
                for(auto* face : first)
                    targets.emplace_back(face);
            }
            frameBufferMap.emplace(node, std::move(targets));
        } else {
            throw Error("Unsupported shader pass type");
        }
    }

    for(const auto* node : order) {
        switch(node->getNodeClass()) {
            case NodeClass::GLSLShader: {
                auto& targets = frameBufferMap.at(node);
                std::vector<Channel> channels;
                if(const auto found = inputs.find(node); found != inputs.end()) {
                    channels.reserve(found->second.size());
                    for(const auto* link : found->second) {
                        const auto texture = textureMap.find(link->start);
                        if(texture == textureMap.end())
                            throw Error("Shader input was not prepared before its consumer");
                        std::optional<Vec2> size;
                        if(const auto knownSize = textureSizeMap.find(link->start); knownSize != textureSizeMap.end())
                            size = knownSize->second;
                        channels.push_back(Channel{ link->slot, texture->second, link->filter, link->wrapMode, size });
                    }
                }

                const auto& shader = dynamic_cast<const GLSLShader&>(*node);
                pipeline->addPass(shader.source, shader.nodeType, targets, std::move(channels), node == directRenderNode);

                if(targets.front().t1) {
                    const auto texType = shader.nodeType == NodeType::CubeMap ? TexType::CubeMap : TexType::Tex2D;
                    textureMap.emplace(
                        node, DoubleBufferedTex{ targets.front().t1->getTexture(), targets.front().t2->getTexture(), texType });
                }
                break;
            }
            case NodeClass::LastFrame: {
                const auto& lastFrame = dynamic_cast<const LastFrame&>(*node);
                const auto target = frameBufferMap.at(lastFrame.refNode).front();
                if(!target.t1 || !target.t2)
                    throw Error("LastFrame source is not double buffered");
                textureMap.emplace(node,
                                   DoubleBufferedTex{ target.t2->getTexture(), target.t1->getTexture(),
                                                      lastFrame.refNode->getNodeType() == NodeType::CubeMap ? TexType::CubeMap :
                                                                                                              TexType::Tex2D });
                break;
            }
            case NodeClass::RenderOutput:
                break;
            case NodeClass::Texture: {
                const auto& texture = dynamic_cast<const Texture&>(*node);
                if(texture.pixel.size() != static_cast<std::size_t>(texture.width) * texture.height)
                    throw Error("Texture node has an invalid pixel payload");
                const auto id = pipeline->createTexture(texture.width, texture.height, texture.pixel.data());
                textureSizeMap.emplace(node, Vec2{ static_cast<float>(texture.width), static_cast<float>(texture.height) });
                textureMap.emplace(node, DoubleBufferedTex{ id, TexType::Tex2D });
                break;
            }
            case NodeClass::CubeMap: {
                const auto& texture = dynamic_cast<const CubeMap&>(*node);
                if(texture.pixel.size() != static_cast<std::size_t>(texture.size) * texture.size * 6U)
                    throw Error("Cubemap node has an invalid pixel payload");
                const auto id = pipeline->createCubeMap(texture.size, texture.pixel.data());
                textureSizeMap.emplace(node, Vec2{ static_cast<float>(texture.size), static_cast<float>(texture.size) });
                textureMap.emplace(node, DoubleBufferedTex{ id, TexType::CubeMap });
                break;
            }
            case NodeClass::Volume: {
                const auto& volume = dynamic_cast<const Volume&>(*node);
                const auto expected = static_cast<std::size_t>(volume.size) * volume.size * volume.size * volume.channels;
                if(volume.pixel.size() != expected)
                    throw Error("Volume node has an invalid voxel payload");
                const auto id = pipeline->createVolume(volume.size, volume.channels, volume.pixel.data());
                textureSizeMap.emplace(node, Vec2{ static_cast<float>(volume.size), static_cast<float>(volume.size) });
                textureMap.emplace(node, DoubleBufferedTex{ id, TexType::Tex3D });
                break;
            }
            case NodeClass::Keyboard: {
                const auto id = pipeline->createKeyboardTexture();
                textureSizeMap.emplace(
                    node, Vec2{ static_cast<float>(KeyboardInput::KeyCount), static_cast<float>(KeyboardInput::Rows) });
                textureMap.emplace(node, DoubleBufferedTex{ id, TexType::Tex2D });
                break;
            }
            case NodeClass::Music: {
                const auto id = pipeline->createAudioTexture();
                textureSizeMap.emplace(
                    node, Vec2{ static_cast<float>(AudioInput::TextureWidth), static_cast<float>(AudioInput::TextureHeight) });
                textureMap.emplace(node, DoubleBufferedTex{ id, TexType::Tex2D });
                break;
            }
            case NodeClass::SoundOutput:
            case NodeClass::Unknown:
                throw Error("Unsupported node class in renderer");
        }
    }

    return pipeline;
}

SHADERTOY_NAMESPACE_END
