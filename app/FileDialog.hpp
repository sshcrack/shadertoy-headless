/*
    SPDX-License-Identifier: Apache-2.0
    Copyright 2023-2026 Yingwei Zheng and contributors
*/
#pragma once

#include <optional>
#include <string>
#include <vector>

namespace ShaderToy::FileDialog {

    void shutdown();

    std::optional<std::string> openFile(const char* filterName, const char* extensions);
    std::optional<std::string> saveFile(const char* filterName, const char* extensions);
    std::vector<std::string> openFiles(const char* filterName, const char* extensions);

}  // namespace ShaderToy::FileDialog
