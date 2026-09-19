/*
    SPDX-License-Identifier: Apache-2.0
    Copyright 2023-2026 Yingwei Zheng and contributors
*/

#include "FileDialog.hpp"

#include <hello_imgui/hello_imgui.h>
#include <nfd.h>

namespace ShaderToy::FileDialog {

    namespace {
        bool initialized = false;

        void logError(const char* operation) {
            const auto* error = NFD_GetError();
            HelloImGui::Log(HelloImGui::LogLevel::Error, "%s failed: %s", operation, error ? error : "unknown error");
        }

        bool ensureInitialized() {
            if(initialized)
                return true;
            if(NFD_Init() != NFD_OKAY) {
                logError("Native file dialog initialization");
                return false;
            }
            initialized = true;
            return true;
        }
    }  // namespace

    void shutdown() {
        if(!initialized)
            return;
        NFD_Quit();
        initialized = false;
    }

    std::optional<std::string> openFile(const char* filterName, const char* extensions) {
        if(!ensureInitialized())
            return std::nullopt;
        const nfdfilteritem_t filters[] = { { filterName, extensions } };
        nfdchar_t* path = nullptr;
        const auto result = NFD_OpenDialog(&path, filters, 1, nullptr);
        if(result == NFD_CANCEL)
            return std::nullopt;
        if(result != NFD_OKAY) {
            logError("Open dialog");
            return std::nullopt;
        }

        std::string value(path);
        NFD_FreePath(path);
        return value;
    }

    std::optional<std::string> saveFile(const char* filterName, const char* extensions) {
        if(!ensureInitialized())
            return std::nullopt;
        const nfdfilteritem_t filters[] = { { filterName, extensions } };
        nfdchar_t* path = nullptr;
        const auto result = NFD_SaveDialog(&path, filters, 1, nullptr, nullptr);
        if(result == NFD_CANCEL)
            return std::nullopt;
        if(result != NFD_OKAY) {
            logError("Save dialog");
            return std::nullopt;
        }

        std::string value(path);
        NFD_FreePath(path);
        return value;
    }

    std::vector<std::string> openFiles(const char* filterName, const char* extensions) {
        if(!ensureInitialized())
            return {};
        const nfdfilteritem_t filters[] = { { filterName, extensions } };
        const nfdpathset_t* paths = nullptr;
        const auto result = NFD_OpenDialogMultiple(&paths, filters, 1, nullptr);
        if(result == NFD_CANCEL)
            return {};
        if(result != NFD_OKAY) {
            logError("Open multiple dialog");
            return {};
        }

        std::vector<std::string> values;
        nfdpathsetsize_t count = 0;
        if(NFD_PathSet_GetCount(paths, &count) != NFD_OKAY) {
            logError("Reading dialog paths");
            NFD_PathSet_Free(paths);
            return {};
        }

        values.reserve(count);
        for(nfdpathsetsize_t index = 0; index < count; ++index) {
            nfdchar_t* path = nullptr;
            if(NFD_PathSet_GetPath(paths, index, &path) != NFD_OKAY) {
                logError("Reading dialog path");
                continue;
            }
            values.emplace_back(path);
            NFD_PathSet_FreePath(path);
        }

        NFD_PathSet_Free(paths);
        return values;
    }

}  // namespace ShaderToy::FileDialog
