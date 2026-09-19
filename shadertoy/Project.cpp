/*
    SPDX-License-Identifier: Apache-2.0
    Copyright 2023-2026 Yingwei Zheng and contributors
*/
#include "shadertoy/Project.hpp"

#include "shadertoy/Error.hpp"

#include <memory>
#include <unordered_map>
#include <unordered_set>

SHADERTOY_NAMESPACE_BEGIN

namespace {
    NodeType passNodeType(const ProjectPassKind kind) {
        switch(kind) {
            case ProjectPassKind::Image:
            case ProjectPassKind::Buffer:
                return NodeType::Image;
            case ProjectPassKind::CubeMap:
                return NodeType::CubeMap;
        }
        throw Error("Unknown project pass kind");
    }
}  // namespace

Result<ShaderDocument> makeProjectDocument(const ProjectDescription& project) {
    try {
        if(project.name.empty())
            throw Error("Project name must not be empty");
        if(project.passes.empty())
            throw Error("Project must contain at least one shader pass");

        ShaderDocument document;
        document.metadata.emplace("Name", project.name);

        std::unordered_map<std::string, Node*> sources;
        std::unordered_map<std::string, GLSLShader*> passes;
        std::unordered_set<std::string> names;
        GLSLShader* finalImage = nullptr;

        for(const auto& texture : project.textures) {
            if(texture.name.empty())
                throw Error("Texture name must not be empty");
            if(!names.emplace(texture.name).second)
                throw Error("Duplicate project source name: " + texture.name);
            if(texture.width == 0 || texture.height == 0)
                throw Error("Texture dimensions must be positive: " + texture.name);
            if(texture.rgba.size() != static_cast<std::size_t>(texture.width) * texture.height)
                throw Error("Texture pixel payload has the wrong size: " + texture.name);

            auto node = std::make_unique<Texture>(texture.width, texture.height, texture.rgba);
            node->name = texture.name;
            sources.emplace(texture.name, node.get());
            document.nodes.emplace_back(std::move(node));
        }

        for(const auto& pass : project.passes) {
            if(pass.name.empty())
                throw Error("Pass name must not be empty");
            if(!names.emplace(pass.name).second)
                throw Error("Duplicate project source name: " + pass.name);
            if(pass.source.empty())
                throw Error("Pass source must not be empty: " + pass.name);

            auto node = std::make_unique<GLSLShader>(pass.source, passNodeType(pass.kind));
            node->name = pass.name;
            auto* shader = node.get();
            passes.emplace(pass.name, shader);
            sources.emplace(pass.name, shader);
            if(pass.kind == ProjectPassKind::Image) {
                if(finalImage)
                    throw Error("Project must contain exactly one image pass");
                finalImage = shader;
            }
            document.nodes.emplace_back(std::move(node));
        }

        if(!finalImage)
            throw Error("Project must contain exactly one image pass");

        Keyboard* keyboard = nullptr;
        Music* music = nullptr;
        uint32_t feedbackIndex = 0;

        for(const auto& pass : project.passes) {
            auto* consumer = passes.at(pass.name);
            std::unordered_set<uint32_t> usedChannels;
            for(const auto& input : pass.inputs) {
                if(input.channel > 3)
                    throw Error("Channel index must be between 0 and 3 in pass " + pass.name);
                if(!usedChannels.emplace(input.channel).second)
                    throw Error("Duplicate channel " + std::to_string(input.channel) + " in pass " + pass.name);

                Node* producer = nullptr;
                switch(input.kind) {
                    case ProjectInputKind::Pass:
                    case ProjectInputKind::Texture: {
                        const auto found = sources.find(input.source);
                        if(found == sources.end())
                            throw Error("Unknown input source '" + input.source + "' in pass " + pass.name);
                        producer = found->second;
                        if(input.kind == ProjectInputKind::Pass && producer->getNodeClass() != NodeClass::GLSLShader)
                            throw Error("Input source is not a shader pass: " + input.source);
                        if(input.kind == ProjectInputKind::Texture && producer->getNodeClass() != NodeClass::Texture)
                            throw Error("Input source is not a texture: " + input.source);
                        break;
                    }
                    case ProjectInputKind::Keyboard:
                        if(input.previousFrame)
                            throw Error("Keyboard inputs cannot use previous-frame semantics");
                        if(!keyboard) {
                            auto node = std::make_unique<Keyboard>();
                            node->name = "keyboard";
                            keyboard = node.get();
                            document.nodes.emplace_back(std::move(node));
                        }
                        producer = keyboard;
                        break;
                    case ProjectInputKind::Music:
                        if(input.previousFrame)
                            throw Error("Music inputs cannot use previous-frame semantics");
                        if(!music) {
                            auto node = std::make_unique<Music>();
                            node->name = "music";
                            music = node.get();
                            document.nodes.emplace_back(std::move(node));
                        }
                        producer = music;
                        break;
                }

                if(input.previousFrame) {
                    if(input.kind != ProjectInputKind::Pass)
                        throw Error("Only shader pass inputs can use previous-frame semantics");
                    auto* sourcePass = dynamic_cast<GLSLShader*>(producer);
                    if(!sourcePass)
                        throw Error("Previous-frame source is not a shader pass: " + input.source);
                    if(sourcePass == finalImage)
                        throw Error("The final image pass cannot be used as a previous-frame source");

                    auto last = std::make_unique<LastFrame>(sourcePass->name, sourcePass->nodeType);
                    last->name = "__feedback_" + std::to_string(feedbackIndex++);
                    last->refNode = sourcePass;
                    producer = last.get();
                    document.nodes.emplace_back(std::move(last));
                }

                document.links.push_back(Link{ producer, consumer, input.filter, input.wrap, input.channel });
            }
        }

        auto output = std::make_unique<RenderOutput>();
        output->name = "Output";
        auto* outputPtr = output.get();
        document.nodes.emplace_back(std::move(output));
        document.links.push_back(Link{ finalImage, outputPtr, Filter::Linear, Wrap::Clamp, 0 });

        return document;
    } catch(const Error& error) {
        return std::unexpected(error);
    } catch(const std::exception& error) {
        return std::unexpected(Error(error.what()));
    }
}

SHADERTOY_NAMESPACE_END
