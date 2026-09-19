/*
    SPDX-License-Identifier: Apache-2.0
    Copyright 2023-2026 Yingwei Zheng and contributors
*/

#include <shadertoy/ShaderToy.hpp>

#include <filesystem>
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

    const auto roundTripPath = std::filesystem::temp_directory_path() / "shadertoy-library-smoke.sttf";
    image->save(roundTripPath.string());

    ShaderToy::ShaderDocument roundTrip;
    roundTrip.load(roundTripPath.string());
    std::filesystem::remove(roundTripPath);
    if(!require(roundTrip.nodes.size() == 2 && roundTrip.links.size() == 1, "STTF round-trip changed document shape"))
        return 1;
    if(!require(roundTrip.metadata.at("Name") == "smoke", "STTF round-trip lost metadata"))
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
