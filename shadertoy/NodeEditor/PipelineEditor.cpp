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

#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
#define CPPHTTPLIB_OPENSSL_SUPPORT
#endif

#define IMGUI_DEFINE_MATH_OPERATORS
#include "shadertoy/NodeEditor/PipelineEditor.hpp"
#include <queue>

#include "shadertoy/SuppressWarningPush.hpp"

#include <fmt/format.h>
#include <hello_imgui/dpi_aware.h>
#include <hello_imgui/hello_imgui.h>
#include <httplib.h>
#include <magic_enum/magic_enum.hpp>
#include <nlohmann/json.hpp>
#include <stb_image.h>

using HelloImGui::EmToVec2;

#include "shadertoy/SuppressWarningPop.hpp"
#include <unordered_set>

SHADERTOY_NAMESPACE_BEGIN

static constexpr auto initialShader = R"(void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;

    // Time varying pixel color
    vec3 col = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(0,2,4));

    // Output to screen
    fragColor = vec4(col,1.0);
}
)";

uint32_t PipelineEditor::nextId() {
    return mNextId++;
}

void PipelineEditor::setupInitialPipeline() {
    auto& shader = spawnShader(NodeType::Image);
    shader.currShaderText = initialShader;
    auto& sink = spawnRenderOutput();

    mLinks.emplace_back(nextId(), shader.outputs.front().id, sink.inputs.front().id);
}

PipelineEditor::PipelineEditor() {
    setupInitialPipeline();
    mShouldBuildPipeline = true;
    mShouldResetLayout = true;
}

void PipelineEditor::resetPipeline() {
    mNodes.clear();
    mLinks.clear();
    mMetadata.clear();
    setupInitialPipeline();
    mShouldBuildPipeline = true;
    mShouldResetLayout = true;
}

PipelineEditor::~PipelineEditor() = default;

bool PipelineEditor::isUniqueName(const std::string_view& name, const EditorNode* exclude) const {
    return std::all_of(mNodes.cbegin(), mNodes.cend(), [&](auto& node) { return node.get() == exclude || node->name != name; });
}
std::string PipelineEditor::generateUniqueName(const std::string_view& base) const {
    if(isUniqueName(base, nullptr))
        return { base.data(), base.size() };
    for(uint32_t idx = 1;; ++idx) {
        if(auto str = fmt::format("{}{}", base, idx); isUniqueName(str, nullptr)) {
            return str;
        }
    }
}

template <typename T>
static auto& buildNode(std::vector<std::unique_ptr<EditorNode>>& nodes, std::unique_ptr<T> node) {
    for(auto& input : node->inputs) {
        input.node = node.get();
        input.kind = PinKind::Input;
    }

    for(auto& output : node->outputs) {
        output.node = node.get();
        output.kind = PinKind::Output;
    }
    auto& ref = *node;
    nodes.push_back(std::move(node));
    return ref;
}
EditorTexture& PipelineEditor::spawnTexture() {
    auto ret = std::make_unique<EditorTexture>(nextId(), generateUniqueName("Texture"));
    ret->outputs.emplace_back(nextId(), "Output", NodeType::Image);
    return buildNode(mNodes, std::move(ret));
}
EditorCubeMap& PipelineEditor::spawnCubeMap() {
    auto ret = std::make_unique<EditorCubeMap>(nextId(), generateUniqueName("CubeMap"));
    ret->type = NodeType::CubeMap;
    ret->outputs.emplace_back(nextId(), "Output", NodeType::CubeMap);
    return buildNode(mNodes, std::move(ret));
}
EditorVolume& PipelineEditor::spawnVolume() {
    auto ret = std::make_unique<EditorVolume>(nextId(), generateUniqueName("Volume"));
    ret->type = NodeType::Volume;
    ret->outputs.emplace_back(nextId(), "Output", NodeType::Volume);
    return buildNode(mNodes, std::move(ret));
}
EditorKeyboard& PipelineEditor::spawnKeyboard() {
    auto ret = std::make_unique<EditorKeyboard>(nextId(), generateUniqueName("Keyboard"));
    ret->outputs.emplace_back(nextId(), "Output", NodeType::Image);
    return buildNode(mNodes, std::move(ret));
}
EditorRenderOutput& PipelineEditor::spawnRenderOutput() {
    auto ret = std::make_unique<EditorRenderOutput>(nextId(), generateUniqueName("RenderOutput"));
    ret->inputs.emplace_back(nextId(), "Input", NodeType::Image);
    return buildNode(mNodes, std::move(ret));
}
EditorLastFrame& PipelineEditor::spawnLastFrame() {
    auto ret = std::make_unique<EditorLastFrame>(nextId(), generateUniqueName("LastFrame"));
    ret->outputs.emplace_back(nextId(), "Output", NodeType::Image);
    return buildNode(mNodes, std::move(ret));
}
EditorShader& PipelineEditor::spawnShader(NodeType type) {
    auto ret = std::make_unique<EditorShader>(nextId(), generateUniqueName("Shader"));
    ret->type = type;
    for(uint32_t idx = 0; idx < 4; ++idx) {
        ret->inputs.emplace_back(nextId(), fmt::format("Channel{}", idx).c_str(), NodeType::Image);
    }
    ret->outputs.emplace_back(nextId(), "Output", type);
    return buildNode(mNodes, std::move(ret));
}

EditorPin* PipelineEditor::findPin(const ed::PinId id) const {
    if(!id)
        return nullptr;

    for(auto& node : mNodes) {
        for(auto& pin : node->inputs)
            if(pin.id == id)
                return &pin;

        for(auto& pin : node->outputs)
            if(pin.id == id)
                return &pin;
    }

    return nullptr;
}

std::unique_ptr<Pipeline> PipelineEditor::buildPipeline() {
    std::unordered_map<EditorNode*, std::vector<std::tuple<EditorNode*, uint32_t, EditorLink*>>> graph;
    EditorNode* directRenderNode = nullptr;
    std::unordered_map<EditorNode*, uint32_t> degree;
    EditorNode* sinkNode = nullptr;
    for(auto& link : mLinks) {
        auto u = findPin(link.startPinId);
        auto v = findPin(link.endPinId);
        auto idx = static_cast<uint32_t>(v - v->node->inputs.data());
        graph[v->node].emplace_back(u->node, idx, &link);
        ++degree[u->node];
        if(v->node->getClass() == NodeClass::RenderOutput && u->node->getClass() == NodeClass::GLSLShader) {
            sinkNode = v->node;
            directRenderNode = u->node;
        }
    }

    if(!sinkNode) {
        std::string msg = "Exactly one shader should be connected to the final render output";
        HelloImGui::Log(HelloImGui::LogLevel::Error, msg.c_str());
        throw std::runtime_error(msg);
    }

    std::unordered_set<EditorNode*> visited;
    std::queue<EditorNode*> q;
    std::vector<EditorNode*> order;
    q.push(sinkNode);
    std::unordered_set<EditorNode*> weakRef;
    for(auto& node : mNodes) {
        if(node->getClass() == NodeClass::LastFrame) {
            weakRef.insert(dynamic_cast<EditorLastFrame*>(node.get())->lastFrame);
        }
    }
    for(auto node : weakRef) {
        if(!degree.count(node))
            q.push(node);
    }
    while(!q.empty()) {
        auto u = q.front();
        q.pop();
        visited.insert(u);
        order.push_back(u);

        if(auto it = graph.find(u); it != graph.cend()) {
            for(auto [v, idx, link] : it->second) {
                visited.insert(v);
                if(--degree[v] == 0) {
                    q.push(v);
                }
            }
        }
    }

    if(visited.size() != order.size()) {
        std::string msg = "Loop detected";
        HelloImGui::Log(HelloImGui::LogLevel::Error, msg.c_str());
        throw std::runtime_error(msg);
    }

    std::reverse(order.begin(), order.end());

    auto pipeline = createPipeline();
    std::unordered_map<EditorNode*, DoubleBufferedTex> textureMap;
    std::unordered_map<EditorNode*, ImVec2> textureSizeMap;
    std::unordered_map<EditorNode*, std::vector<DoubleBufferedFB>> frameBufferMap;
    std::unordered_set<EditorNode*> requireDoubleBuffer;
    for(auto node : order) {
        if(node->getClass() == NodeClass::LastFrame) {
            auto ref = dynamic_cast<EditorLastFrame&>(*node).lastFrame;
            if(!ref || ref == directRenderNode) {
                std::string msg = "Invalid reference";
                HelloImGui::Log(HelloImGui::LogLevel::Error, msg.c_str());
                throw std::runtime_error(msg);
            }
            requireDoubleBuffer.insert(ref);
        }
    }
    // TODO: FB allocation
    for(auto node : order) {
        if(node->getClass() == NodeClass::GLSLShader) {
            if(node->type == NodeType::Image) {
                DoubleBufferedFB frameBuffer{ nullptr };
                if(requireDoubleBuffer.count(node)) {
                    auto t1 = pipeline->createFrameBuffer();
                    auto t2 = pipeline->createFrameBuffer();
                    frameBuffer = DoubleBufferedFB{ t1, t2 };
                } else if(node != directRenderNode) {
                    auto t = pipeline->createFrameBuffer();
                    frameBuffer = DoubleBufferedFB{ t };
                }
                frameBufferMap.emplace(node, std::vector<DoubleBufferedFB>{ frameBuffer });
            } else if(node->type == NodeType::CubeMap) {
                std::vector<DoubleBufferedFB> buffers;
                buffers.reserve(6);
                if(requireDoubleBuffer.count(node)) {
                    auto t1 = pipeline->createCubeMapFrameBuffer();
                    auto t2 = pipeline->createCubeMapFrameBuffer();
                    for(uint32_t idx = 0; idx < 6; ++idx)
                        buffers.emplace_back(t1[idx], t2[idx]);
                } else {
                    assert(node != directRenderNode);
                    auto t = pipeline->createCubeMapFrameBuffer();
                    for(uint32_t idx = 0; idx < 6; ++idx)
                        buffers.emplace_back(t[idx]);
                }
                frameBufferMap.emplace(node, std::move(buffers));
            } else {
                std::string msg = "Unsupported shader type" ;
                HelloImGui::Log(HelloImGui::LogLevel::Error, msg.c_str());
                throw std::runtime_error(msg);
            }
        }
    }

    for(auto node : order) {
        switch(node->getClass()) {  // NOLINT(clang-diagnostic-switch-enum)
            case NodeClass::GLSLShader: {
                auto& target = frameBufferMap.at(node);
                std::vector<Channel> channels;
                if(auto it = graph.find(node); it != graph.cend()) {
                    for(auto [v, idx, link] : it->second) {
                        std::optional<ImVec2> size = std::nullopt;
                        if(auto iter = textureSizeMap.find(v); iter != textureSizeMap.cend())
                            size = iter->second;
                        channels.push_back(Channel{ idx, textureMap.at(v), link->filter, link->wrapMode, size });
                    }
                }
                // TODO: error markers
                auto guard = scopeFail(
                    [&] { HelloImGui::Log(HelloImGui::LogLevel::Error, "Failed to compile shader %s", node->name.c_str()); });
                pipeline->addPass(dynamic_cast<EditorShader*>(node)->currShaderText, node->type, target, std::move(channels),
                                  node == sinkNode);
                if(target.front().t1)
                    textureMap.emplace(node,
                                       DoubleBufferedTex{ target.front().t1->getTexture(), target.front().t2->getTexture(),
                                                          node->type == NodeType::CubeMap ? TexType::CubeMap : TexType::Tex2D });
                break;
            }
            case NodeClass::LastFrame: {
                const auto ref = dynamic_cast<EditorLastFrame*>(node)->lastFrame;
                auto target = frameBufferMap.at(ref).front();
                assert(target.t1 && target.t2);
                textureMap.emplace(node,
                                   DoubleBufferedTex{ target.t2->getTexture(), target.t1->getTexture(),
                                                      ref->type == NodeType::CubeMap ? TexType::CubeMap : TexType::Tex2D });
                break;
            }
            case NodeClass::RenderOutput: {
                break;
            }
            case NodeClass::Texture: {
                auto& textureId = dynamic_cast<EditorTexture*>(node)->textureId;
                textureSizeMap.emplace(node, textureId->size());
                textureMap.emplace(node, DoubleBufferedTex{ textureId->getTexture(), TexType::Tex2D });
                break;
            }
            case NodeClass::CubeMap: {
                auto& textureId = dynamic_cast<EditorCubeMap*>(node)->textureId;
                textureSizeMap.emplace(node, textureId->size());
                textureMap.emplace(node, DoubleBufferedTex{ textureId->getTexture(), TexType::CubeMap });
                break;
            }
            case NodeClass::Volume: {
                auto& textureId = dynamic_cast<EditorVolume*>(node)->textureId;
                textureSizeMap.emplace(node, textureId->size());
                textureMap.emplace(node, DoubleBufferedTex{ textureId->getTexture(), TexType::Tex3D });
                break;
            }
            default: {
                std::string msg = "Not implemented node class in buildPipeline";
                //reportNotImplemented();
                throw std::runtime_error(msg);
            }
        }
    }

    return pipeline;
}

std::expected<void, std::runtime_error> PipelineEditor::build(ShaderToyContext &context) {
    try {
        const auto start = Clock::now();
        context.reset(buildPipeline());
        const auto duration =
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count()) * 1e-9;
        Log(HelloImGui::LogLevel::Info, "Compiled in %.1f secs", duration);

        return {};
    } catch(const std::runtime_error& e) {
        return std::unexpected(e);
    }
}

std::expected<void, std::runtime_error> PipelineEditor::update(ShaderToyContext &context) {
    if(mShouldBuildPipeline) {
        auto res = build(context);
        mShouldBuildPipeline = false;

        return res;
    }

    return {};
}

PipelineEditor& PipelineEditor::get() {
    static PipelineEditor instance;
    return instance;
}

std::unique_ptr<Node> EditorRenderOutput::toSTTF() const {
    return std::make_unique<RenderOutput>();
}
void EditorRenderOutput::fromSTTF(Node& node) {
    type = node.getNodeType();
}

std::unique_ptr<Node> EditorShader::toSTTF() const {
    return std::make_unique<GLSLShader>(currShaderText, type);
}
void EditorShader::fromSTTF(Node& node) {
    const auto& shader = dynamic_cast<GLSLShader&>(node);
    type = shader.nodeType;
    currShaderText = shader.source;
}

std::unique_ptr<Node> EditorTexture::toSTTF() const {
    return std::make_unique<Texture>(static_cast<uint32_t>(textureId->size().x), static_cast<uint32_t>(textureId->size().y),
                                     pixel);
}
void EditorTexture::fromSTTF(Node& node) {
    const auto& texture = dynamic_cast<Texture&>(node);
    pixel = texture.pixel;
    textureId = loadTexture(texture.width, texture.height, pixel.data());
}
std::unique_ptr<Node> EditorCubeMap::toSTTF() const {
    return std::make_unique<CubeMap>(static_cast<uint32_t>(textureId->size().x), pixel);
}
void EditorCubeMap::fromSTTF(Node& node) {
    const auto& texture = dynamic_cast<CubeMap&>(node);
    pixel = texture.pixel;
    textureId = loadCubeMap(texture.size, pixel.data());
}

std::unique_ptr<Node> EditorVolume::toSTTF() const {
    const auto size = static_cast<uint32_t>(textureId->size().x);
    const auto channels = pixel.size() / (size * size * size);
    return std::make_unique<Volume>(static_cast<uint32_t>(textureId->size().x), static_cast<uint32_t>(channels), pixel);
}
void EditorVolume::fromSTTF(Node& node) {
    const auto& texture = dynamic_cast<Volume&>(node);
    pixel = texture.pixel;
    textureId = loadVolume(texture.size, texture.channels, pixel.data());
}

std::unique_ptr<Node> EditorLastFrame::toSTTF() const {
    return std::make_unique<LastFrame>(lastFrame->name, type);
}
void EditorLastFrame::fromSTTF(Node&) {
    // should be fixed by post processing
}

void PipelineEditor::loadFromShaderToy(const std::string& path) {
    std::vector<std::unique_ptr<EditorNode>> oldNodes;
    oldNodes.swap(mNodes);
    std::vector<EditorLink> oldLinks;
    oldLinks.swap(mLinks);
    std::vector<std::pair<std::string, std::string>> oldMetadata;
    oldMetadata.swap(mMetadata);
    auto guard = scopeFail([&] {
        oldNodes.swap(mNodes);
        oldLinks.swap(mLinks);
        oldMetadata.swap(mMetadata);
    });

    std::string_view shaderId = path;
    if(const auto pos = shaderId.find_last_of('/'); pos != std::string_view::npos)
        shaderId = shaderId.substr(pos + 1);
    const auto url = fmt::format("https://www.shadertoy.com/view/{}", shaderId);
    HelloImGui::Log(HelloImGui::LogLevel::Info, "Loading from %s", url.c_str());
    httplib::SSLClient client{ "www.shadertoy.com" };
    httplib::Headers headers;
    headers.emplace("referer", url);
    auto res = client.Post("/shadertoy", headers, std::string(R"(s={"shaders":[")") + shaderId.data() + "\"]}&nt=1&nl=1&np=1",
                           "application/x-www-form-urlencoded");
    int status = res.value().status;
    if(status != 200) {
        std::string msg = fmt::format("Invalid response from shadertoy.com (Status code = %d).", status);
        HelloImGui::Log(HelloImGui::LogLevel::Error, msg.c_str());
        throw std::runtime_error(msg);
    }
    auto json = nlohmann::json::parse(res->body);
    if(!json.is_array()) {
        std::string msg = "Invalid response from shadertoy.com";
        HelloImGui::Log(HelloImGui::LogLevel::Error, msg.c_str());
        throw std::runtime_error(msg);
    }
    auto metadata = json[0].at("info");

    mMetadata.emplace_back("Name", metadata.at("name").get<std::string>());
    mMetadata.emplace_back("Author", metadata.at("username").get<std::string>());
    mMetadata.emplace_back("Description", metadata.at("description").get<std::string>());
    mMetadata.emplace_back("ShaderToyURL", url);

    auto renderPasses = json[0].at("renderpass");
    // BA BB BC BD CA IE
    auto getOrder = [](const std::string& name) { return std::toupper(name.front()) * 1000 + std::toupper(name.back()); };

    std::unordered_map<std::string, EditorShader*> newShaderNodes;

    auto& sinkNode = spawnRenderOutput();
    auto addLink = [&](EditorNode* src, EditorNode* dst, uint32_t channel, nlohmann::json* ref) {
        auto filter = Filter::Linear;
        auto wrapMode = Wrap::Repeat;
        if(ref) {
            auto sampler = ref->at("sampler");
            const auto filterName = sampler.at("filter").get<std::string>();
            const auto wrapName = sampler.at("wrap").get<std::string>();
            if(filterName == "linear") {
                filter = Filter::Linear;
            } else if(filterName == "nearest") {
                filter = Filter::Nearest;
            } else if(filterName == "mipmap") {
                filter = Filter::Mipmap;
            } else {
                reportNotImplemented();
            }

            if(wrapName == "clamp") {
                wrapMode = Wrap::Clamp;
            } else if(wrapName == "repeat") {
                wrapMode = Wrap::Repeat;
            } else {
                reportNotImplemented();
            }
        }
        mLinks.emplace_back(nextId(), src->outputs.front().id, dst->inputs[channel].id, filter, wrapMode);
    };
    EditorNode* keyboard = nullptr;
    auto getKeyboard = [&] {
        if(!keyboard)
            keyboard = &spawnKeyboard();
        return keyboard;
    };
    std::unordered_map<std::string, EditorTexture*> textureCache;
    auto getTexture = [&](nlohmann::json& tex) -> EditorTexture* {
        const auto id = tex.at("id").get<std::string>();
        if(const auto iter = textureCache.find(id); iter != textureCache.cend())
            return iter->second;
        auto& texture = spawnTexture();
        const auto texPath = tex.at("filepath").get<std::string>();
        HelloImGui::Log(HelloImGui::LogLevel::Info, "Downloading texture %s", texPath.c_str());
        auto img = client.Get(texPath, headers);

        stbi_set_flip_vertically_on_load(tex.at("sampler").at("vflip").get<std::string>() == "true");
        int width, height, channels;
        const auto ptr = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(img->body.data()),
                                               static_cast<int>(img->body.size()), &width, &height, &channels, 4);
        if(!ptr) {
            std::string msg = fmt::format("Failed to load texture %s: %s", texPath.c_str(), stbi_failure_reason());
            HelloImGui::Log(HelloImGui::LogLevel::Info, msg.c_str());
            throw std::runtime_error(msg);
        }
        const auto imgGuard = scopeExit([ptr] { stbi_image_free(ptr); });
        const auto begin = reinterpret_cast<const uint32_t*>(ptr);
        const auto end = begin + static_cast<ptrdiff_t>(width) * height;
        texture.pixel = std::vector<uint32_t>{ begin, end };
        texture.textureId = loadTexture(static_cast<uint32_t>(width), static_cast<uint32_t>(height), texture.pixel.data());

        textureCache.emplace(id, &texture);
        return &texture;
    };
    std::unordered_map<std::string, EditorCubeMap*> cubeMapCache;
    auto getCubeMap = [&](nlohmann::json& tex) -> EditorCubeMap* {
        const auto id = tex.at("id").get<std::string>();
        if(const auto iter = cubeMapCache.find(id); iter != cubeMapCache.cend())
            return iter->second;
        auto& texture = spawnCubeMap();
        const auto texPath = tex.at("filepath").get<std::string>();
        std::string base, ext;
        if(const auto pos = texPath.find_last_of('.'); pos != std::string::npos) {
            base = texPath.substr(0, pos);
            ext = texPath.substr(pos);
        } else {
            std::string msg = fmt::format("Failed to parse cube map %s", texPath.c_str());
            HelloImGui::Log(HelloImGui::LogLevel::Info, msg.c_str());
            throw std::runtime_error(msg);
        }

        constexpr const char* suffixes[] = { "", "_1", "_2", "_3", "_4", "_5" };
        int32_t size = 0;
        for(const auto suffix : suffixes) {
            auto facePath = base;
            facePath += suffix;
            facePath += ext;
            HelloImGui::Log(HelloImGui::LogLevel::Info, "Downloading texture %s", facePath.c_str());
            auto img = client.Get(facePath, headers);

            stbi_set_flip_vertically_on_load(tex.at("sampler").at("vflip").get<std::string>() == "true");
            int width, height, channels;
            const auto ptr = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(img->body.data()),
                                                   static_cast<int>(img->body.size()), &width, &height, &channels, 4);
            if(!ptr) {
                std::string msg = fmt::format("Failed to load texture %s: %s", facePath.c_str(), stbi_failure_reason());
                HelloImGui::Log(HelloImGui::LogLevel::Info, msg.c_str());
                throw std::runtime_error(msg);
            }
            const auto imgGuard = scopeExit([ptr] { stbi_image_free(ptr); });
            const auto begin = reinterpret_cast<const uint32_t*>(ptr);
            const auto end = begin + static_cast<ptrdiff_t>(width) * height;
            texture.pixel.insert(texture.pixel.end(), begin, end);
            if(width != height) {
                std::string msg = "Cube map face width != height";
                throw std::runtime_error(msg);
            }
            if(size == 0)
                size = width;
            else if(size != width) {
                std::string msg = "Cube map face size mismatch";
                throw std::runtime_error(msg);
            }
        }

        texture.textureId = loadCubeMap(static_cast<uint32_t>(size), texture.pixel.data());
        cubeMapCache.emplace(id, &texture);
        return &texture;
    };
    std::unordered_map<std::string, EditorVolume*> volumeCache;
    auto getVolume = [&](nlohmann::json& tex) -> EditorVolume* {
        const auto id = tex.at("id").get<std::string>();
        if(const auto iter = volumeCache.find(id); iter != volumeCache.cend())
            return iter->second;
        auto& texture = spawnVolume();
        const auto texPath = tex.at("filepath").get<std::string>();
        HelloImGui::Log(HelloImGui::LogLevel::Info, "Downloading volume %s", texPath.c_str());
        auto img = client.Get(texPath, headers);
        if(img->body.empty()) {
            std::string msg = fmt::format("Failed to load texture %s: %s", texPath.c_str(), stbi_failure_reason());
            HelloImGui::Log(HelloImGui::LogLevel::Info, msg.c_str());
            throw std::runtime_error(msg);
        }
        if(img->body.size() < 20ULL) {
            std::string msg = fmt::format("Invalid volume format %s: %zu", texPath.c_str(), img->body.size());
            HelloImGui::Log(HelloImGui::LogLevel::Info, msg.c_str());
            throw std::runtime_error(msg);
        }
        auto begin = reinterpret_cast<const uint32_t*>(img->body.data());
        uint32_t x = *++begin;
        uint32_t y = *++begin;
        uint32_t z = *++begin;
        if(x != y || y != z) {
            std::string msg = fmt::format("Unsupported volume size %s: (%u, %u, %u)", texPath.c_str(), x, y, z);
            HelloImGui::Log(HelloImGui::LogLevel::Info, msg.c_str());
            throw std::runtime_error(msg);
        }
        uint32_t channels_layout_format = *++begin;
        struct Metadata final {
            uint8_t channels;
            uint8_t layout;
            uint16_t format;
        } metadata;
        static_assert(sizeof(metadata) == sizeof(uint32_t));
        memcpy(&metadata, &channels_layout_format, sizeof(Metadata));

        if(metadata.channels != 1 && metadata.channels != 4) {
            std::string msg = fmt::format("Unsupported volume channels %s: %u", texPath.c_str(), metadata.channels);
            HelloImGui::Log(HelloImGui::LogLevel::Info, msg.c_str());
            throw std::runtime_error(msg);
        }

        if(metadata.layout != 0) {
            std::string msg = fmt::format("Unsupported volume layout %s: %u", texPath.c_str(), metadata.layout);
            HelloImGui::Log(HelloImGui::LogLevel::Info, msg.c_str());
            throw std::runtime_error(msg);
        }

        if(metadata.format != 0) {
            std::string msg = fmt::format("Unsupported volume format %s: %u", texPath.c_str(), metadata.format);
            HelloImGui::Log(HelloImGui::LogLevel::Info, msg.c_str());
            throw std::runtime_error(msg);
        }

        const uint32_t size = x;
        const uint32_t channels = metadata.channels;
        const size_t points = static_cast<size_t>(size) * size * size * channels;
        if(img->body.size() != 20 + points) {
            std::string msg = fmt::format("Invalid volume format %s: %zu", texPath.c_str(), img->body.size());
            HelloImGui::Log(HelloImGui::LogLevel::Info, msg.c_str());
            throw std::runtime_error(msg);
        }
        const auto start = img->body.data() + 20;
        const auto end = start + points;
        texture.pixel = std::vector<uint8_t>{ start, end };
        texture.textureId = loadVolume(size, channels, texture.pixel.data());

        volumeCache.emplace(id, &texture);
        return &texture;
    };
    std::unordered_set<std::string> passIds;
    const auto isDynamicCubeMap = [&](nlohmann::json& tex) {
        const auto id = tex.at("id").get<std::string>();
        return passIds.count(id) != 0;
    };
    std::string common;
    for(auto& pass : renderPasses) {
        if(pass.at("name").get<std::string>().empty()) {
            pass.at("name") = generateUniqueName(pass.at("type").get<std::string>());
        }
        if(pass.at("outputs").empty()) {
            pass.at("outputs").push_back(nlohmann::json::object({ { "id", "tmp" + std::to_string(nextId()) } }));
        }
        passIds.insert(pass.at("outputs")[0].at("id").get<std::string>());
    }
    for(auto& pass : renderPasses) {
        const auto type = pass.at("type").get<std::string>();
        const auto code = pass.at("code").get<std::string>();
        const auto name = pass.at("name").get<std::string>();
        if(type == "common") {
            common = code + '\n';
        } else if(type == "image" || type == "buffer" || type == "cubemap") {
            const auto output = pass.at("outputs")[0].at("id").get<std::string>();
            auto& node = spawnShader(type != "cubemap" ? NodeType::Image : NodeType::CubeMap);
            node.currShaderText = code;
            node.name = name;
            newShaderNodes.emplace(output, &node);

            for(auto& input : pass.at("inputs")) {
                auto inputType = input.at("type").get<std::string>();
                if(inputType == "buffer") {
                    continue;
                }
                auto channel = input.at("channel").get<uint32_t>();
                if(inputType == "keyboard") {
                    //addLink(getKeyboard(), &node, channel, &input);
                } else if(inputType == "texture") {
                    addLink(getTexture(input), &node, channel, &input);
                } else if(inputType == "cubemap") {
                    if(!isDynamicCubeMap(input))
                        addLink(getCubeMap(input), &node, channel, &input);
                } else if(inputType == "volume") {
                    addLink(getVolume(input), &node, channel, &input);
                } else {
                    Log(HelloImGui::LogLevel::Error, "Unsupported input type %s", inputType.c_str());
                }
            }

            if(type == "image") {
                addLink(&node, &sinkNode, 0, nullptr);
            }
        } else {
            Log(HelloImGui::LogLevel::Error, "Unsupported pass type %s", type.c_str());
        }
    }

    if(!common.empty()) {
        for(auto& [name, shader] : newShaderNodes) {
            shader->currShaderText = common + shader->currShaderText;
        }
    }

    std::unordered_map<EditorShader*, EditorLastFrame*> lastFrames;
    auto getLastFrame = [&](EditorShader* src) {
        if(const auto iter = lastFrames.find(src); iter != lastFrames.cend()) {
            return iter->second;
        }
        auto& lastFrame = spawnLastFrame();
        lastFrame.lastFrame = src;
        lastFrames.emplace(src, &lastFrame);
        return &lastFrame;
    };
    for(auto& pass : renderPasses) {
        const auto type = pass.at("type").get<std::string>();
        if(type == "common") {
            continue;
        }
        if(type == "image" || type == "buffer" || type == "cubemap") {
            const auto name = pass.at("name").get<std::string>();
            const auto idxDst = getOrder(name);
            const auto node = newShaderNodes.at(pass.at("outputs")[0].at("id").get<std::string>());

            for(auto& input : pass.at("inputs")) {
                auto inputType = input.at("type").get<std::string>();
                if(!(inputType == "buffer" || (inputType == "cubemap" && isDynamicCubeMap(input)))) {
                    continue;
                }

                auto channel = input.at("channel").get<uint32_t>();
                auto src = newShaderNodes.at(input.at("id").get<std::string>());
                const auto idxSrc = getOrder(src->name);
                if(idxSrc < idxDst) {
                    addLink(src, node, channel, &input);
                } else {
                    addLink(getLastFrame(src), node, channel, &input);
                }
            }
        } else {
            Log(HelloImGui::LogLevel::Error, "Unsupported pass type %s", type.c_str());
        }
    }

    mShouldResetLayout = true;
    mShouldBuildPipeline = true;
}

std::string PipelineEditor::getShaderName() const {
    using namespace std::string_view_literals;
    for(auto [k, v] : mMetadata)
        if(k == "Name"sv || k == "name"sv)
            return v;
    return "untitled";
}


SHADERTOY_NAMESPACE_END
