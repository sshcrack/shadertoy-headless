/*
    SPDX-License-Identifier: Apache-2.0
    Copyright 2023-2026 Yingwei Zheng and contributors
*/

#include "shadertoy/STTF.hpp"
#include "shadertoy/Support.hpp"

#include <cstring>
#include <fstream>
#include <unordered_map>

#include "shadertoy/SuppressWarningPush.hpp"
#include <cpp-base64/base64.h>
#include <gsl/gsl>
#include <magic_enum/magic_enum.hpp>
#include <nlohmann/json.hpp>
#include "shadertoy/SuppressWarningPop.hpp"

SHADERTOY_NAMESPACE_BEGIN

namespace {
    template <typename Enum>
    Enum parseEnum(const nlohmann::json& value, const std::string_view field) {
        const auto text = value.get<std::string>();
        if(const auto parsed = magic_enum::enum_cast<Enum>(text))
            return *parsed;
        throw Error("Unknown " + std::string(field) + ": " + text);
    }
}  // namespace

void ShaderToyTransmissionFormat::load(const std::string& filePath) {
    std::ifstream file{ filePath };
    if(!file)
        throw Error("Cannot open STTF file: " + filePath);

    try {
        nlohmann::json json;
        file >> json;

        ShaderToyTransmissionFormat parsed;
        json.at("metadata").get_to(parsed.metadata);

        std::unordered_map<std::string, Node*> nodeMap;
        for(const auto& node : json.at("nodes")) {
            std::unique_ptr<Node> nodeValue;
            switch(parseEnum<NodeClass>(node.at("class"), "node class")) {
                case NodeClass::RenderOutput:
                    nodeValue = std::make_unique<RenderOutput>();
                    break;
                case NodeClass::GLSLShader:
                    nodeValue = std::make_unique<GLSLShader>(node.at("source").get<std::string>(),
                                                             parseEnum<NodeType>(node.at("type"), "node type"));
                    break;
                case NodeClass::Texture: {
                    const auto width = node.at("width").get<uint32_t>();
                    const auto height = node.at("height").get<uint32_t>();
                    if(width == 0 || height == 0)
                        throw Error("Texture dimensions must be positive");
                    const auto decoded = base64_decode(node.at("data").get<std::string>());
                    const auto pixelCount = checkedSizeProduct({ width, height }, "Texture");
                    const auto expectedBytes = checkedSizeProduct({ pixelCount, sizeof(uint32_t) }, "Texture payload");
                    if(decoded.size() != expectedBytes)
                        throw Error("Texture payload has an invalid size");
                    std::vector<uint32_t> pixels(pixelCount);
                    std::memcpy(pixels.data(), decoded.data(), expectedBytes);
                    nodeValue = std::make_unique<Texture>(width, height, std::move(pixels));
                    break;
                }
                case NodeClass::CubeMap: {
                    const auto size = node.at("size").get<uint32_t>();
                    if(size == 0)
                        throw Error("Cubemap size must be positive");
                    const auto decoded = base64_decode(node.at("data").get<std::string>());
                    const auto pixelCount = checkedSizeProduct({ size, size, 6U }, "Cubemap");
                    const auto expectedBytes = checkedSizeProduct({ pixelCount, sizeof(uint32_t) }, "Cubemap payload");
                    if(decoded.size() != expectedBytes)
                        throw Error("Cubemap payload has an invalid size");
                    std::vector<uint32_t> pixels(pixelCount);
                    std::memcpy(pixels.data(), decoded.data(), expectedBytes);
                    nodeValue = std::make_unique<CubeMap>(size, std::move(pixels));
                    break;
                }
                case NodeClass::Volume: {
                    const auto size = node.at("size").get<uint32_t>();
                    const auto channels = node.at("channels").get<uint32_t>();
                    if(size == 0)
                        throw Error("Volume size must be positive");
                    if(channels != 1 && channels != 4)
                        throw Error("Volume channel count must be 1 or 4");
                    const auto decoded = base64_decode(node.at("data").get<std::string>());
                    const auto byteCount = checkedSizeProduct({ size, size, size, channels }, "Volume");
                    if(decoded.size() != byteCount)
                        throw Error("Volume payload has an invalid size");
                    nodeValue = std::make_unique<Volume>(
                        size, channels,
                        std::vector<uint8_t>{ reinterpret_cast<const uint8_t*>(decoded.data()),
                                              reinterpret_cast<const uint8_t*>(decoded.data()) + byteCount });
                    break;
                }
                case NodeClass::LastFrame:
                    nodeValue = std::make_unique<LastFrame>(node.at("ref").get<std::string>(),
                                                            parseEnum<NodeType>(node.at("type"), "node type"));
                    break;
                case NodeClass::Keyboard:
                    nodeValue = std::make_unique<Keyboard>();
                    break;
                case NodeClass::Music:
                    nodeValue = std::make_unique<Music>();
                    break;
                case NodeClass::SoundOutput:
                case NodeClass::Unknown:
                    throw Error("Unsupported node class in STTF file");
            }

            nodeValue->name = node.at("name").get<std::string>();
            if(!nodeMap.emplace(nodeValue->name, nodeValue.get()).second)
                throw Error("Duplicate node name in STTF file: " + nodeValue->name);
            parsed.nodes.push_back(std::move(nodeValue));
        }

        for(auto& node : parsed.nodes) {
            if(node->getNodeClass() != NodeClass::LastFrame)
                continue;
            auto& lastFrame = dynamic_cast<LastFrame&>(*node);
            const auto referenced = nodeMap.find(lastFrame.refNodeName);
            if(referenced == nodeMap.end())
                throw Error("Unknown LastFrame reference: " + lastFrame.refNodeName);
            lastFrame.refNode = referenced->second;
        }

        for(const auto& link : json.at("links")) {
            const auto startName = link.at("start").get<std::string>();
            const auto endName = link.at("end").get<std::string>();
            const auto start = nodeMap.find(startName);
            const auto end = nodeMap.find(endName);
            if(start == nodeMap.end() || end == nodeMap.end())
                throw Error("Link references an unknown node");
            parsed.links.push_back(Link{
                start->second,
                end->second,
                parseEnum<Filter>(link.at("filter"), "filter"),
                parseEnum<Wrap>(link.at("wrapMode"), "wrap mode"),
                link.at("slot").get<uint32_t>(),
            });
        }

        *this = std::move(parsed);
    } catch(const Error&) {
        throw;
    } catch(const std::exception& ex) {
        throw Error("Failed to parse STTF file '" + filePath + "': " + ex.what());
    }
}

void ShaderToyTransmissionFormat::save(const std::string& filePath) const {
    std::ofstream file{ filePath };
    if(!file)
        throw Error("Cannot open STTF file for writing: " + filePath);

    try {
        nlohmann::json json;
        nlohmann::to_json(json["metadata"], metadata);
        auto& jsonNodes = json["nodes"];

        for(const auto& node : nodes) {
            nlohmann::json jsonNode;
            jsonNode["class"] = magic_enum::enum_name(node->getNodeClass());
            jsonNode["name"] = node->name;

            switch(node->getNodeClass()) {
                case NodeClass::RenderOutput:
                case NodeClass::Keyboard:
                case NodeClass::Music:
                case NodeClass::SoundOutput:
                    break;
                case NodeClass::GLSLShader: {
                    const auto& shader = dynamic_cast<const GLSLShader&>(*node);
                    jsonNode["source"] = shader.source;
                    jsonNode["type"] = magic_enum::enum_name(shader.nodeType);
                    break;
                }
                case NodeClass::Texture: {
                    const auto& texture = dynamic_cast<const Texture&>(*node);
                    const auto bytes = gsl::as_bytes(gsl::span<const uint32_t>{ texture.pixel.data(), texture.pixel.size() });
                    jsonNode["data"] = base64_encode(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
                    jsonNode["width"] = texture.width;
                    jsonNode["height"] = texture.height;
                    break;
                }
                case NodeClass::CubeMap: {
                    const auto& texture = dynamic_cast<const CubeMap&>(*node);
                    const auto bytes = gsl::as_bytes(gsl::span<const uint32_t>{ texture.pixel.data(), texture.pixel.size() });
                    jsonNode["data"] = base64_encode(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
                    jsonNode["size"] = texture.size;
                    break;
                }
                case NodeClass::Volume: {
                    const auto& volume = dynamic_cast<const Volume&>(*node);
                    const auto bytes = gsl::as_bytes(gsl::span<const uint8_t>{ volume.pixel.data(), volume.pixel.size() });
                    jsonNode["data"] = base64_encode(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
                    jsonNode["size"] = volume.size;
                    jsonNode["channels"] = volume.channels;
                    break;
                }
                case NodeClass::LastFrame: {
                    const auto& lastFrame = dynamic_cast<const LastFrame&>(*node);
                    jsonNode["ref"] = lastFrame.refNodeName;
                    jsonNode["type"] = magic_enum::enum_name(lastFrame.nodeType);
                    break;
                }
                case NodeClass::Unknown:
                    throw Error("Cannot save an unknown node class");
            }
            jsonNodes.push_back(std::move(jsonNode));
        }

        auto& jsonLinks = json["links"];
        for(const auto& [start, end, filter, wrapMode, slot] : links) {
            if(!start || !end)
                throw Error("Cannot save a link with a null endpoint");
            nlohmann::json jsonLink;
            jsonLink["start"] = start->name;
            jsonLink["end"] = end->name;
            jsonLink["filter"] = magic_enum::enum_name(filter);
            jsonLink["wrapMode"] = magic_enum::enum_name(wrapMode);
            jsonLink["slot"] = slot;
            jsonLinks.push_back(std::move(jsonLink));
        }

        file << json;
    } catch(const Error&) {
        throw;
    } catch(const std::exception& ex) {
        throw Error("Failed to write STTF file '" + filePath + "': " + ex.what());
    }
}

SHADERTOY_NAMESPACE_END
