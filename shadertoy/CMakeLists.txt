find_package(fmt CONFIG REQUIRED)
find_package(GLEW CONFIG REQUIRED)
find_package(Microsoft.GSL CONFIG REQUIRED)
find_package(magic_enum CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(httplib CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_path(CPP_BASE64_INCLUDE_DIRS "cpp-base64/base64.cpp" REQUIRED)

set(SHADERTOY_LIBRARY_SOURCES
    Compiler.cpp
    Importer.cpp
    OpenGL.cpp
    ShaderToy.cpp
    ShaderToyContext.cpp
    STTF.cpp
    ${CPP_BASE64_INCLUDE_DIRS}/cpp-base64/base64.cpp
)

add_library(shadertoy-lib ${SHADERTOY_LIBRARY_SOURCES})
add_library(shadertoy::shadertoy ALIAS shadertoy-lib)
set_target_properties(shadertoy-lib PROPERTIES
    OUTPUT_NAME shadertoy
    EXPORT_NAME shadertoy
)

target_compile_features(shadertoy-lib PUBLIC cxx_std_23)
target_include_directories(shadertoy-lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${Stb_INCLUDE_DIR}
        ${CPP_BASE64_INCLUDE_DIRS}
)
target_link_libraries(shadertoy-lib
    PRIVATE
        fmt::fmt
        GLEW::GLEW
        Microsoft.GSL::GSL
        magic_enum::magic_enum
        httplib::httplib
        OpenSSL::SSL
        OpenSSL::Crypto
        nlohmann_json::nlohmann_json
)

if(MSVC)
    target_compile_options(shadertoy-lib PRIVATE
        /Wall /WX
        /wd4514 /wd4820 /wd4623 /wd4625 /wd4626 /wd5026 /wd5027
        /wd4061 /wd5045 /wd4365 /wd4868 /wd4710 /wd5262 /wd4711 /wd4738
        /D_CRT_SECURE_NO_WARNINGS /D_SILENCE_ALL_MS_EXT_DEPRECATION_WARNINGS
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(shadertoy-lib PRIVATE
        -Wall -Wextra -Werror -Wconversion -Wshadow=compatible-local
        -Wno-psabi -Wno-array-bounds
    )
endif()

set(SHADERTOY_PUBLIC_HEADERS
    AudioInput.hpp
    Config.hpp
    Error.hpp
    Importer.hpp
    Result.hpp
    ShaderToy.hpp
    STTF.hpp
    Types.hpp
)
install(FILES ${SHADERTOY_PUBLIC_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/shadertoy)

install(TARGETS shadertoy-lib
    EXPORT shadertoy-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(EXPORT shadertoy-targets
    FILE shadertoy-targets.cmake
    NAMESPACE shadertoy::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/shadertoy
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/shadertoy-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/shadertoy-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/shadertoy
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/shadertoy-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/shadertoy-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/shadertoy-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/shadertoy
)
