/*
    SPDX-License-Identifier: Apache-2.0
    Copyright 2023-2026 Yingwei Zheng and contributors
*/

#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
#define CPPHTTPLIB_OPENSSL_SUPPORT
#endif

#include "shadertoy/Importer.hpp"
#include "shadertoy/AudioInput.hpp"
#include "shadertoy/Support.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "shadertoy/SuppressWarningPush.hpp"
#include <fmt/format.h>
#include <httplib.h>
#include <nlohmann/json.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "shadertoy/SuppressWarningPop.hpp"
#include <stb_image.h>

SHADERTOY_NAMESPACE_BEGIN

namespace {

    constexpr auto InitialBuffer = R"(void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    fragColor = vec4(0.0,0.0,1.0,1.0);
}
)";

    constexpr auto InitialCubeMap = R"(void mainCubemap( out vec4 fragColor, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir )
{
    vec3 col = 0.5 + 0.5*rayDir;
    fragColor = vec4(col,1.0);
}
)";

    std::string uniqueName(const ShaderDocument& document, std::string base) {
        auto exists = [&](const std::string_view candidate) {
            return std::any_of(document.nodes.begin(), document.nodes.end(),
                               [&](const auto& node) { return node->name == candidate; });
        };

        if(base.empty())
            base = "Node";
        if(!exists(base))
            return base;

        for(uint32_t index = 1;; ++index) {
            auto candidate = fmt::format("{}{}", base, index);
            if(!exists(candidate))
                return candidate;
        }
    }

    template <typename T>
    T* appendNode(ShaderDocument& document, std::unique_ptr<T> node, std::string name) {
        node->name = uniqueName(document, std::move(name));
        auto* result = node.get();
        document.nodes.push_back(std::move(node));
        return result;
    }

    void appendLink(ShaderDocument& document, Node* start, Node* end, const uint32_t slot, const Filter filter = Filter::Linear,
                    const Wrap wrap = Wrap::Repeat) {
        if(!start || !end)
            throw Error("Cannot create a shader link with a null endpoint");
        document.links.push_back(Link{ start, end, filter, wrap, slot });
    }

    std::pair<Filter, Wrap> samplerSettings(const nlohmann::json* input) {
        if(!input)
            return { Filter::Linear, Wrap::Repeat };

        const auto& sampler = input->at("sampler");
        const auto filterName = sampler.at("filter").get<std::string>();
        const auto wrapName = sampler.at("wrap").get<std::string>();

        Filter filter{};
        if(filterName == "linear")
            filter = Filter::Linear;
        else if(filterName == "nearest")
            filter = Filter::Nearest;
        else if(filterName == "mipmap")
            filter = Filter::Mipmap;
        else
            throw Error("Unsupported ShaderToy sampler filter: " + filterName);

        Wrap wrap{};
        if(wrapName == "clamp")
            wrap = Wrap::Clamp;
        else if(wrapName == "repeat")
            wrap = Wrap::Repeat;
        else
            throw Error("Unsupported ShaderToy sampler wrap mode: " + wrapName);

        return { filter, wrap };
    }

    void appendSamplerLink(ShaderDocument& document, Node* start, Node* end, const uint32_t slot, const nlohmann::json* input) {
        const auto [filter, wrap] = samplerSettings(input);
        appendLink(document, start, end, slot, filter, wrap);
    }

    std::string shaderIdFrom(std::string_view value) {
        if(const auto query = value.find_first_of("?#"); query != std::string_view::npos)
            value = value.substr(0, query);
        while(!value.empty() && value.back() == '/')
            value.remove_suffix(1);
        if(const auto slash = value.find_last_of('/'); slash != std::string_view::npos)
            value = value.substr(slash + 1);
        if(value.empty())
            throw Error("ShaderToy shader id is empty");
        return std::string(value);
    }

    httplib::Result download(httplib::SSLClient& client, const std::string& path, const httplib::Headers& headers) {
        auto result = client.Get(path, headers);
        if(!result)
            throw Error("Failed to download ShaderToy resource: " + path);
        if(result->status != 200)
            throw Error(fmt::format("ShaderToy resource '{}' returned HTTP {}", path, result->status));
        return result;
    }

    std::vector<uint32_t> decodeTexture(const std::string& body, const bool verticalFlip, uint32_t& widthOut,
                                        uint32_t& heightOut) {
        stbi_set_flip_vertically_on_load(verticalFlip ? 1 : 0);
        int width = 0;
        int height = 0;
        int channels = 0;
        const auto* bytes = reinterpret_cast<const stbi_uc*>(body.data());
        auto* ptr = stbi_load_from_memory(bytes, static_cast<int>(body.size()), &width, &height, &channels, 4);
        if(!ptr)
            throw Error(std::string("Failed to decode ShaderToy texture: ") +
                        (stbi_failure_reason() ? stbi_failure_reason() : "unknown stb_image error"));
        const auto guard = scopeExit([ptr] { stbi_image_free(ptr); });

        if(width <= 0 || height <= 0)
            throw Error("ShaderToy texture has invalid dimensions");

        widthOut = static_cast<uint32_t>(width);
        heightOut = static_cast<uint32_t>(height);
        const auto* begin = reinterpret_cast<const uint32_t*>(ptr);
        return { begin, begin + static_cast<std::ptrdiff_t>(width) * height };
    }

}  // namespace

Result<ShaderDocument> makeImageShader(std::string name, std::string source, const std::optional<uint32_t> audioChannel) {
    try {
        if(audioChannel && *audioChannel >= 4)
            throw Error("Audio channel must be between 0 and 3");

        ShaderDocument document;
        document.metadata.emplace("Name", name);

        auto* shader =
            appendNode(document, std::make_unique<GLSLShader>(std::move(source), NodeType::Image), name.empty() ? "Image" : name);
        auto* output = appendNode(document, std::make_unique<RenderOutput>(), "RenderOutput");
        appendLink(document, shader, output, 0);

        if(audioChannel) {
            auto* music = appendNode(document, std::make_unique<Music>(), "Music");
            appendLink(document, music, shader, *audioChannel, Filter::Linear, Wrap::Clamp);
        }

        return document;
    } catch(const Error& error) {
        return std::unexpected(error);
    } catch(const std::exception& error) {
        return std::unexpected(Error(error.what()));
    }
}

Result<ShaderDocument> importFromShaderToy(const std::string_view shaderUrlOrId) {
    try {
        const auto shaderId = shaderIdFrom(shaderUrlOrId);
        const auto url = fmt::format("https://www.shadertoy.com/view/{}", shaderId);

        httplib::SSLClient client{ "www.shadertoy.com" };
        httplib::Headers headers;
        headers.emplace("referer", url);

        const auto form = std::string("s={\"shaders\":[\"") + shaderId + "\"]}&nt=1&nl=1&np=1";
        auto response = client.Post("/shadertoy", headers, form, "application/x-www-form-urlencoded");
        if(!response)
            throw Error("Failed to connect to shadertoy.com");
        if(response->status != 200)
            throw Error(fmt::format("shadertoy.com returned HTTP {}", response->status));

        return importFromShaderToyResponse(shaderId, response->body);
    } catch(const Error& error) {
        return std::unexpected(error);
    } catch(const std::exception& error) {
        return std::unexpected(Error(error.what()));
    }
}

Result<ShaderDocument> importFromShaderToyResponse(const std::string_view shaderIdView, const std::string_view responseBody) {
    try {
        const std::string shaderId{ shaderIdView };
        const auto url = fmt::format("https://www.shadertoy.com/view/{}", shaderId);

        auto json = nlohmann::json::parse(responseBody);
        if(!json.is_array() || json.empty())
            throw Error("Invalid response from shadertoy.com");

        ShaderDocument document;
        const auto& metadata = json[0].at("info");
        document.metadata.emplace("Name", metadata.value("name", std::string{}));
        document.metadata.emplace("Author", metadata.value("username", std::string{}));
        document.metadata.emplace("Description", metadata.value("description", std::string{}));
        document.metadata.emplace("ShaderToyURL", url);

        auto renderPasses = json[0].at("renderpass");
        uint32_t syntheticId = 0;
        for(auto& pass : renderPasses) {
            auto name = pass.value("name", std::string{});
            if(name.empty())
                pass["name"] = pass.value("type", std::string("pass")) + std::to_string(++syntheticId);
            if(pass.at("outputs").empty())
                pass.at("outputs").push_back(nlohmann::json::object({ { "id", "tmp" + std::to_string(++syntheticId) } }));
        }

        auto* sink = appendNode(document, std::make_unique<RenderOutput>(), "RenderOutput");
        std::unordered_map<std::string, GLSLShader*> shaderNodes;

        httplib::SSLClient client{ "www.shadertoy.com" };
        httplib::Headers headers;
        headers.emplace("referer", url);

        std::unordered_map<std::string, Texture*> textureCache;
        std::unordered_map<std::string, CubeMap*> cubeMapCache;
        std::unordered_map<std::string, Volume*> volumeCache;
        Keyboard* keyboard = nullptr;
        Music* music = nullptr;

        auto getKeyboard = [&]() -> Keyboard* {
            if(!keyboard)
                keyboard = appendNode(document, std::make_unique<Keyboard>(), "Keyboard");
            return keyboard;
        };
        auto getMusic = [&]() -> Music* {
            if(!music)
                music = appendNode(document, std::make_unique<Music>(), "Music");
            return music;
        };

        auto getTexture = [&](const nlohmann::json& input) -> Texture* {
            const auto id = input.at("id").get<std::string>();
            if(const auto found = textureCache.find(id); found != textureCache.end())
                return found->second;

            const auto path = input.at("filepath").get<std::string>();
            const auto image = download(client, path, headers);
            uint32_t width = 0;
            uint32_t height = 0;
            const auto flip = input.at("sampler").value("vflip", std::string("false")) == "true";
            auto pixels = decodeTexture(image->body, flip, width, height);
            auto* node = appendNode(document, std::make_unique<Texture>(width, height, std::move(pixels)), "Texture");
            textureCache.emplace(id, node);
            return node;
        };

        auto getCubeMap = [&](const nlohmann::json& input) -> CubeMap* {
            const auto id = input.at("id").get<std::string>();
            if(const auto found = cubeMapCache.find(id); found != cubeMapCache.end())
                return found->second;

            const auto path = input.at("filepath").get<std::string>();
            const auto dot = path.find_last_of('.');
            if(dot == std::string::npos)
                throw Error("Failed to parse cubemap path: " + path);
            const auto base = path.substr(0, dot);
            const auto extension = path.substr(dot);
            constexpr const char* suffixes[] = { "", "_1", "_2", "_3", "_4", "_5" };

            std::vector<uint32_t> pixels;
            uint32_t faceSize = 0;
            const auto flip = input.at("sampler").value("vflip", std::string("false")) == "true";
            for(const auto* suffix : suffixes) {
                const auto facePath = base + suffix + extension;
                const auto image = download(client, facePath, headers);
                uint32_t width = 0;
                uint32_t height = 0;
                auto face = decodeTexture(image->body, flip, width, height);
                if(width != height)
                    throw Error("Cubemap face width does not match height");
                if(faceSize == 0)
                    faceSize = width;
                else if(faceSize != width)
                    throw Error("Cubemap faces have inconsistent sizes");
                pixels.insert(pixels.end(), face.begin(), face.end());
            }

            auto* node = appendNode(document, std::make_unique<CubeMap>(faceSize, std::move(pixels)), "CubeMap");
            cubeMapCache.emplace(id, node);
            return node;
        };

        auto getVolume = [&](const nlohmann::json& input) -> Volume* {
            const auto id = input.at("id").get<std::string>();
            if(const auto found = volumeCache.find(id); found != volumeCache.end())
                return found->second;

            const auto path = input.at("filepath").get<std::string>();
            const auto response = download(client, path, headers);
            const auto& body = response->body;
            if(body.size() < 20)
                throw Error("Invalid ShaderToy volume payload");

            auto readU32 = [&](const std::size_t offset) {
                uint32_t value = 0;
                std::memcpy(&value, body.data() + offset, sizeof(value));
                return value;
            };

            const auto x = readU32(4);
            const auto y = readU32(8);
            const auto z = readU32(12);
            if(x != y || y != z)
                throw Error("Only cubic ShaderToy volumes are supported");

            struct Metadata final {
                uint8_t channels;
                uint8_t layout;
                uint16_t format;
            } volumeMetadata{};
            const auto packed = readU32(16);
            std::memcpy(&volumeMetadata, &packed, sizeof(volumeMetadata));
            if(volumeMetadata.channels != 1 && volumeMetadata.channels != 4)
                throw Error("Unsupported ShaderToy volume channel count");
            if(volumeMetadata.layout != 0 || volumeMetadata.format != 0)
                throw Error("Unsupported ShaderToy volume layout/format");

            const auto pointBytes = static_cast<std::size_t>(x) * x * x * static_cast<std::size_t>(volumeMetadata.channels);
            if(body.size() != 20 + pointBytes)
                throw Error("ShaderToy volume payload has an invalid size");

            const auto* begin = reinterpret_cast<const uint8_t*>(body.data() + 20);
            std::vector<uint8_t> pixels(begin, begin + pointBytes);
            auto* node = appendNode(document, std::make_unique<Volume>(x, volumeMetadata.channels, std::move(pixels)), "Volume");
            volumeCache.emplace(id, node);
            return node;
        };

        const auto isDynamicCubeMap = [](const nlohmann::json& input) { return input.value("id", std::string{}) == "4dX3Rr"; };

        std::string common;
        for(const auto& pass : renderPasses) {
            const auto type = pass.at("type").get<std::string>();
            if(type == "common")
                common = pass.at("code").get<std::string>() + '\n';
        }

        // First create all render passes and static resources.
        for(const auto& pass : renderPasses) {
            const auto type = pass.at("type").get<std::string>();
            if(type == "common")
                continue;
            if(type != "image" && type != "buffer" && type != "cubemap")
                continue;

            const auto outputId = pass.at("outputs")[0].at("id").get<std::string>();
            const auto nodeType = type == "cubemap" ? NodeType::CubeMap : NodeType::Image;
            const auto name = pass.at("name").get<std::string>();
            auto* shader =
                appendNode(document, std::make_unique<GLSLShader>(common + pass.at("code").get<std::string>(), nodeType), name);
            shaderNodes.emplace(outputId, shader);

            for(const auto& input : pass.at("inputs")) {
                const auto inputType = input.at("type").get<std::string>();
                const auto channel = input.at("channel").get<uint32_t>();
                if(channel >= 4)
                    throw Error("ShaderToy channel index is outside iChannel0..3");

                if(inputType == "buffer")
                    continue;
                if(inputType == "keyboard")
                    appendSamplerLink(document, getKeyboard(), shader, channel, &input);
                else if(inputType == "music" || inputType == "musicstream" || inputType == "mic" || inputType == "audio")
                    appendSamplerLink(document, getMusic(), shader, channel, &input);
                else if(inputType == "texture")
                    appendSamplerLink(document, getTexture(input), shader, channel, &input);
                else if(inputType == "cubemap") {
                    if(!isDynamicCubeMap(input))
                        appendSamplerLink(document, getCubeMap(input), shader, channel, &input);
                } else if(inputType == "volume")
                    appendSamplerLink(document, getVolume(input), shader, channel, &input);
                // Unsupported ShaderToy inputs intentionally remain unbound,
                // matching the previous renderer's tolerant import behavior.
            }

            if(type == "image")
                appendLink(document, shader, sink, 0);
        }

        std::unordered_map<GLSLShader*, LastFrame*> lastFrames;
        auto getLastFrame = [&](GLSLShader* source) {
            if(const auto found = lastFrames.find(source); found != lastFrames.end())
                return found->second;
            auto lastFrame = std::make_unique<LastFrame>(source->name, source->nodeType);
            lastFrame->refNode = source;
            auto* node = appendNode(document, std::move(lastFrame), source->name + " LastFrame");
            lastFrames.emplace(source, node);
            return node;
        };

        const auto passOrder = [](const std::string& name) {
            if(name.empty())
                return 0;
            return std::toupper(static_cast<unsigned char>(name.front())) * 1000 +
                std::toupper(static_cast<unsigned char>(name.back()));
        };

        // Then wire buffer/dynamic-cubemap dependencies. Back edges become
        // LastFrame references, which is the ShaderToy feedback-buffer model.
        for(const auto& pass : renderPasses) {
            const auto type = pass.at("type").get<std::string>();
            if(type == "common" || (type != "image" && type != "buffer" && type != "cubemap"))
                continue;

            auto* destination = shaderNodes.at(pass.at("outputs")[0].at("id").get<std::string>());
            const auto destinationOrder = passOrder(pass.at("name").get<std::string>());

            for(const auto& input : pass.at("inputs")) {
                const auto inputType = input.at("type").get<std::string>();
                const bool dynamicCubeMap = inputType == "cubemap" && isDynamicCubeMap(input);
                if(inputType != "buffer" && !dynamicCubeMap)
                    continue;

                auto channel = input.at("channel").get<uint32_t>();
                auto inputId = input.at("id").get<std::string>();

                if(dynamicCubeMap) {
                    for(const auto& candidate : renderPasses) {
                        if(candidate.at("type").get<std::string>() != "cubemap" || candidate.at("outputs").empty())
                            continue;
                        inputId = candidate.at("outputs")[0].at("id").get<std::string>();
                        channel = candidate.at("outputs")[0].value("channel", channel);
                        break;
                    }
                }

                GLSLShader* source = nullptr;
                if(const auto found = shaderNodes.find(inputId); found != shaderNodes.end()) {
                    source = found->second;
                } else {
                    const auto sourceType = inputType == "cubemap" ? NodeType::CubeMap : NodeType::Image;
                    source =
                        appendNode(document,
                                   std::make_unique<GLSLShader>(
                                       common + (sourceType == NodeType::CubeMap ? InitialCubeMap : InitialBuffer), sourceType),
                                   inputId);
                    shaderNodes.emplace(inputId, source);
                }

                Node* linkSource = source;
                if(passOrder(source->name) >= destinationOrder)
                    linkSource = getLastFrame(source);
                appendSamplerLink(document, linkSource, destination, channel, &input);
            }
        }

        return document;
    } catch(const Error& error) {
        return std::unexpected(error);
    } catch(const std::exception& error) {
        return std::unexpected(Error(error.what()));
    }
}

SHADERTOY_NAMESPACE_END
