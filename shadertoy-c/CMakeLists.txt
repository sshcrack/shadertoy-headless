if(UNIX AND NOT APPLE)
    find_path(EGL_REGISTRY_INCLUDE_DIRS "EGL/egl.h" REQUIRED)
else()
    find_package(glfw3 CONFIG REQUIRED)
endif()
if(SHADERTOY_BUILD_C_API_STATIC)
    find_package(fmt CONFIG REQUIRED)
    find_package(glad CONFIG REQUIRED)
    find_package(OpenSSL REQUIRED)
    find_package(unofficial-brotli CONFIG REQUIRED)
endif()

function(configure_shadertoy_c_target target)
    target_compile_features(${target} PRIVATE cxx_std_23)
    target_link_libraries(${target} PRIVATE shadertoy::shadertoy)
    if(UNIX AND NOT APPLE)
        target_include_directories(${target} PRIVATE ${EGL_REGISTRY_INCLUDE_DIRS})
    else()
        target_link_libraries(${target} PRIVATE glfw)
    endif()
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

    if(MSVC)
        target_compile_options(${target} PRIVATE /Wall /WX /EHsc /wd4514 /wd4820 /wd4623 /wd4625 /wd4626 /wd5026 /wd5027 /wd4061 /wd5045 /wd4365 /wd4868 /wd4710 /wd5262 /wd4711 /wd4738)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${target} PRIVATE -Wall -Wextra -Werror -Wconversion -Wshadow=compatible-local)
    endif()
endfunction()

if(SHADERTOY_BUILD_C_API)
    add_library(shadertoy-c SHARED src/shadertoy.cpp)
    add_library(shadertoy::c ALIAS shadertoy-c)
    configure_shadertoy_c_target(shadertoy-c)
    target_compile_definitions(shadertoy-c PRIVATE SHADERTOY_C_BUILD)
    set_target_properties(shadertoy-c PROPERTIES
        OUTPUT_NAME shadertoy_c
        EXPORT_NAME c
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
    )
    install(TARGETS shadertoy-c
        EXPORT shadertoy-targets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

if(SHADERTOY_BUILD_C_API_STATIC)
    add_library(shadertoy-c-static STATIC src/shadertoy.cpp)
    add_library(shadertoy::c-static ALIAS shadertoy-c-static)
    configure_shadertoy_c_target(shadertoy-c-static)
    target_compile_definitions(shadertoy-c-static PUBLIC SHADERTOY_C_STATIC)
    set_target_properties(shadertoy-c-static PROPERTIES
        OUTPUT_NAME shadertoy_c_static
        EXPORT_NAME c-static
        POSITION_INDEPENDENT_CODE ON
    )
    # Cargo owns the final executable link. Keep the native dependency graph in
    # CMake and generate a tiny machine-readable manifest for shadertoy-sys
    # rather than duplicating vcpkg library names in Rust build.rs.
    set(SHADERTOY_RUST_SYSTEM_LINKS "")
    if(WIN32)
        string(APPEND SHADERTOY_RUST_SYSTEM_LINKS
            "system=opengl32\n"
            "system=gdi32\n"
            "system=user32\n"
            "system=shell32\n"
            "system=advapi32\n"
            "system=crypt32\n"
            "system=ws2_32\n"
            "system=bcrypt\n"
            "system=ncrypt\n"
            "system=secur32\n"
            "system=userenv\n"
            "system=dwmapi\n"
        )
    elseif(APPLE)
        string(APPEND SHADERTOY_RUST_SYSTEM_LINKS
            "system=c++\n"
            "framework=OpenGL\n"
            "framework=Cocoa\n"
            "framework=IOKit\n"
            "framework=CoreFoundation\n"
            "framework=QuartzCore\n"
            "framework=Security\n"
        )
    else()
        string(APPEND SHADERTOY_RUST_SYSTEM_LINKS
            "system=stdc++\n"
            "system=m\n"
            "system=rt\n"
            "system=dl\n"
            "system=pthread\n"
        )
    endif()

    set(SHADERTOY_RUST_CONTEXT_STATIC_LINK "")
    if(NOT (UNIX AND NOT APPLE))
        set(SHADERTOY_RUST_CONTEXT_STATIC_LINK "static=$<TARGET_LINKER_FILE:glfw>\n")
    endif()

    file(GENERATE
        OUTPUT "${CMAKE_BINARY_DIR}/shadertoy-rust-link-$<CONFIG>.txt"
        CONTENT
"static=$<TARGET_LINKER_FILE:shadertoy-c-static>
static=$<TARGET_LINKER_FILE:shadertoy-lib>
${SHADERTOY_RUST_CONTEXT_STATIC_LINK}static=$<TARGET_LINKER_FILE:fmt::fmt>
static=$<TARGET_LINKER_FILE:glad::glad>
static=$<TARGET_LINKER_FILE:unofficial::brotli::brotlienc>
static=$<TARGET_LINKER_FILE:unofficial::brotli::brotlidec>
static=$<TARGET_LINKER_FILE:unofficial::brotli::brotlicommon>
static=$<TARGET_LINKER_FILE:OpenSSL::SSL>
static=$<TARGET_LINKER_FILE:OpenSSL::Crypto>
${SHADERTOY_RUST_SYSTEM_LINKS}"
    )
endif()

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(BUILD_TESTING)
    enable_language(C)
    if(TARGET shadertoy-c)
        set(SHADERTOY_C_SMOKE_TARGET shadertoy-c)
    elseif(TARGET shadertoy-c-static)
        set(SHADERTOY_C_SMOKE_TARGET shadertoy-c-static)
    endif()
    if(SHADERTOY_C_SMOKE_TARGET)
        add_executable(shadertoy-c-smoke tests/smoke.c)
        target_link_libraries(shadertoy-c-smoke PRIVATE ${SHADERTOY_C_SMOKE_TARGET})
        add_test(NAME shadertoy-c-smoke COMMAND shadertoy-c-smoke)
    endif()
endif()
