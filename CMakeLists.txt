cmake_minimum_required(VERSION 3.10)

project(shadertoy-headless)

include(GNUInstallDirs)

add_subdirectory(shadertoy)

option(SHADERTOY_BUILD_PREVIEW_TOOL "Build the deterministic hidden-window shader preview CLI" OFF)
if(SHADERTOY_BUILD_PREVIEW_TOOL)
    find_package(glfw3 CONFIG REQUIRED)
    find_package(GLEW REQUIRED)
    find_package(Stb REQUIRED)
    add_executable(shadertoy-preview tools/shadertoy-preview.cpp)
    target_compile_features(shadertoy-preview PRIVATE cxx_std_23)
    target_include_directories(shadertoy-preview PRIVATE ${Stb_INCLUDE_DIR})
    target_link_libraries(shadertoy-preview PRIVATE shadertoy-headless glfw GLEW::GLEW)
endif()
