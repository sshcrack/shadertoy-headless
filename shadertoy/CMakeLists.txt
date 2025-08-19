cmake_minimum_required(VERSION 3.10)

project(shadertoy-headless)

include(GNUInstallDirs)

# Find dependencies
find_package(fmt CONFIG REQUIRED)
find_package(GLEW CONFIG REQUIRED)
find_package(Microsoft.GSL CONFIG REQUIRED)
find_package(magic_enum CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(httplib CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)
find_path(CPP_BASE64_INCLUDE_DIRS "cpp-base64/base64.cpp")
find_package(nlohmann_json 3.2.0 REQUIRED)
if(NOT TARGET hello-imgui::hello_imgui)
    find_package(hello-imgui CONFIG REQUIRED)
endif()

file(GLOB SHADERTOY_SRC "*.cpp" "NodeEditor/*.cpp")
file(GLOB SHADERTOY_HEADERS "*.hpp" "NodeEditor/*.hpp")

# Remove any entry in SHADERTOY_SRC ending with /shadertoy.cpp
set(SHADERTOY_SRC_TO_REMOVE "")
foreach(src_file ${SHADERTOY_SRC})
    string(REGEX MATCH "/shadertoy\\.cpp$" match ${src_file})
    if(match)
        set(SHADERTOY_SRC_TO_REMOVE ${src_file})
        break()
    endif()
endforeach()
if(SHADERTOY_SRC_TO_REMOVE)
    list(REMOVE_ITEM SHADERTOY_SRC "${SHADERTOY_SRC_TO_REMOVE}")
endif()

add_library(shadertoy-headless ${SHADERTOY_SRC} ${SHADERTOY_HEADERS} ${CPP_BASE64_INCLUDE_DIRS}/cpp-base64/base64.cpp)

# Link libraries
 target_link_libraries(shadertoy-headless PRIVATE fmt::fmt GLEW::GLEW Microsoft.GSL::GSL magic_enum::magic_enum httplib::httplib OpenSSL::SSL OpenSSL::Crypto hello-imgui::hello_imgui nlohmann_json::nlohmann_json)

# Include directories
 target_include_directories(shadertoy-headless PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> 
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${Stb_INCLUDE_DIR} ${CPP_BASE64_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Compile features and options
 target_compile_features(shadertoy-headless PRIVATE cxx_std_23)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set_source_files_properties(${SHADERTOY_SRC} PROPERTIES COMPILE_FLAGS "/Wall /WX /wd4514 /wd4820 /wd4623 /wd4625 /wd4626 /wd5026 /wd5027 /wd4061 /wd5045 /wd4365 /wd4868 /wd4710 /wd5262 /wd4711 /wd4738 /D_CRT_SECURE_NO_WARNINGS /D_SILENCE_ALL_MS_EXT_DEPRECATION_WARNINGS")
elseif(CMAKE_COMPILER_IS_GNUCXX)
    set_source_files_properties(${SHADERTOY_SRC} PROPERTIES COMPILE_FLAGS "-Wall -Wextra -Werror -Wconversion -Wshadow=compatible-local -Wno-psabi -Wno-array-bounds -Wno-unused-parameter")
endif()

# Install the library and its headers
install(TARGETS shadertoy-headless
        EXPORT shadertoy-headless_targets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

install(FILES ${SHADERTOY_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/shadertoy)

# Export targets
install(EXPORT shadertoy-headless_targets
        FILE shadertoy-headless-targets.cmake
        NAMESPACE shadertoy_headless::
        DESTINATION share/shadertoy-headless)

# Generate the config file
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/shadertoy-headless-config.cmake"
"include(CMakeFindDependencyMacro)\n"
"find_dependency(fmt CONFIG REQUIRED)\n"
"find_dependency(GLEW CONFIG REQUIRED)\n"
"find_dependency(Microsoft.GSL CONFIG REQUIRED)\n"
"find_dependency(magic_enum CONFIG REQUIRED)\n"
"find_dependency(Stb REQUIRED)\n"
"find_dependency(httplib CONFIG REQUIRED)\n"
"find_dependency(OpenSSL REQUIRED)\n"
"find_dependency(nlohmann_json 3.2.0 REQUIRED)\n"
"find_dependency(hello-imgui CONFIG REQUIRED)\n"
"include(\"${CMAKE_CURRENT_LIST_DIR}/shadertoy-headless-targets.cmake\")\n"
)

# Install the generated config file
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/shadertoy-headless-config.cmake"
        DESTINATION share/shadertoy-headless)
