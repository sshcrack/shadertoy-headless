/*
    SPDX-License-Identifier: Apache-2.0
    Copyright 2023-2026 Yingwei Zheng and contributors
*/

#include <shadertoy/ShaderToy.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

    bool require(const bool condition, const char* message) {
        if(condition)
            return true;
        std::cerr << message << '\n';
        return false;
    }

}  // namespace

int main() {
    auto image = ShaderToy::makeImageShader(
        "smoke", R"(void mainImage(out vec4 color, in vec2 coord) { color = vec4(coord / iResolution.xy, 0.0, 1.0); })");
    if(!require(image.has_value(), "makeImageShader failed"))
        return 1;
    if(!require(image->nodes.size() == 2 && image->links.size() == 1, "unexpected one-pass document shape"))
        return 1;
    ShaderToy::CustomUniformValue stormUniform;
    stormUniform.type = ShaderToy::CustomUniformType::Float;
    stormUniform.value.x = 0.75f;
    image->uniforms.emplace("u_storm", stormUniform);

    const auto roundTripPath = std::filesystem::temp_directory_path() / "shadertoy-library-smoke.sttf";
    image->save(roundTripPath.string());

    ShaderToy::ShaderDocument roundTrip;
    roundTrip.load(roundTripPath.string());
    std::filesystem::remove(roundTripPath);
    if(!require(roundTrip.nodes.size() == 2 && roundTrip.links.size() == 1, "STTF round-trip changed document shape"))
        return 1;
    if(!require(roundTrip.metadata.at("Name") == "smoke", "STTF round-trip lost metadata"))
        return 1;
    if(!require(roundTrip.uniforms.contains("u_storm"), "STTF round-trip lost custom uniform"))
        return 1;
    if(!require(roundTrip.uniforms.at("u_storm").type == ShaderToy::CustomUniformType::Float &&
                    roundTrip.uniforms.at("u_storm").value.x == 0.75f,
                "STTF round-trip changed custom uniform value"))
        return 1;

    const auto overflowPath = std::filesystem::temp_directory_path() / "shadertoy-library-overflow.sttf";
    {
        std::ofstream file(overflowPath);
        file << R"json({
          "metadata": {},
          "nodes": [
            { "class": "Texture", "name": "huge", "width": 2147483648, "height": 2147483648, "data": "" }
          ],
          "links": []
        })json";
    }
    bool rejectedOverflow = false;
    try {
        ShaderToy::ShaderDocument malformed;
        malformed.load(overflowPath.string());
    } catch(const ShaderToy::Error&) {
        rejectedOverflow = true;
    }
    std::filesystem::remove(overflowPath);
    if(!require(rejectedOverflow, "STTF size overflow was not rejected"))
        return 1;

    const auto channelPath = std::filesystem::temp_directory_path() / "shadertoy-library-volume-channels.sttf";
    {
        std::ofstream file(channelPath);
        file << R"json({
          "metadata": {},
          "nodes": [
            { "class": "Volume", "name": "bad-volume", "size": 1, "channels": 2, "data": "AAA=" }
          ],
          "links": []
        })json";
    }
    bool rejectedChannels = false;
    try {
        ShaderToy::ShaderDocument malformed;
        malformed.load(channelPath.string());
    } catch(const ShaderToy::Error&) {
        rejectedChannels = true;
    }
    std::filesystem::remove(channelPath);
    if(!require(rejectedChannels, "unsupported STTF volume channels were not rejected"))
        return 1;

    const auto duplicateStoragePath = std::filesystem::temp_directory_path() / "shadertoy-library-duplicate-storage-binding.sttf";
    {
        std::ofstream file(duplicateStoragePath);
        file << R"json({
          "metadata": {},
          "nodes": [
            {
              "class": "GLSLShader",
              "name": "Image",
              "type": "Image",
              "source": "void mainImage(out vec4 c, in vec2 p) { c = vec4(0.0); }",
              "storageBuffers": [
                { "name": "first", "binding": 2, "size": 16 },
                { "name": "second", "binding": 2, "size": 16 }
              ]
            }
          ],
          "links": []
        })json";
    }
    bool rejectedDuplicateStorageBinding = false;
    try {
        ShaderToy::ShaderDocument malformed;
        malformed.load(duplicateStoragePath.string());
    } catch(const ShaderToy::Error&) {
        rejectedDuplicateStorageBinding = true;
    }
    std::filesystem::remove(duplicateStoragePath);
    if(!require(rejectedDuplicateStorageBinding, "duplicate STTF storage binding was not rejected"))
        return 1;

    constexpr std::string_view Response = R"json([
      {
        "info": {
          "name": "Audio smoke",
          "username": "test",
          "description": "offline importer fixture"
        },
        "renderpass": [
          {
            "name": "Image",
            "type": "image",
            "code": "void mainImage(out vec4 c, in vec2 p) { c = vec4(texture(iChannel0, vec2(0.5)).rrr, 1.0); }",
            "inputs": [
              {
                "id": "music",
                "type": "music",
                "channel": 0,
                "sampler": { "filter": "linear", "wrap": "clamp", "vflip": "false" }
              }
            ],
            "outputs": [ { "id": "Image" } ]
          }
        ]
      }
    ])json";

    auto imported = ShaderToy::importFromShaderToyResponse("abc123", Response);
    if(!require(imported.has_value(), "offline ShaderToy response import failed"))
        return 1;

    bool hasMusic = false;
    for(const auto& node : imported->nodes)
        hasMusic |= node->getNodeClass() == ShaderToy::NodeClass::Music;
    if(!require(hasMusic, "audio ShaderToy input did not create a Music node"))
        return 1;

    ShaderToy::Runtime runtime;
    ShaderToy::AudioInput audio;
    audio.available = true;
    runtime.setAudioInput(audio);
    if(!require(!runtime.isValid(), "empty runtime should not have a compiled pipeline"))
        return 1;

    return 0;
}
