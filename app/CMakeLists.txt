find_package(fmt CONFIG REQUIRED)
find_package(GLEW CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(hello-imgui CONFIG REQUIRED)
find_package(nfd CONFIG REQUIRED)
find_package(unofficial-imgui-node-editor CONFIG REQUIRED)
find_package(magic_enum CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(httplib CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

file(GLOB SHADERTOY_GUI_SOURCES CONFIGURE_DEPENDS
    "*.cpp"
    "NodeEditor/*.cpp"
)
hello_imgui_add_app(shadertoy
    ${SHADERTOY_GUI_SOURCES}
    ${CMAKE_SOURCE_DIR}/shadertoy/thirdparty/ImGuiColorTextEdit/TextEditor.cpp
)

target_compile_features(shadertoy PRIVATE cxx_std_23)
target_include_directories(shadertoy PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/shadertoy/thirdparty
    ${Stb_INCLUDE_DIR}
)
target_link_libraries(shadertoy PRIVATE
    shadertoy::shadertoy
    fmt::fmt
    GLEW::GLEW
    glfw
    nfd::nfd
    unofficial::imgui-node-editor::imgui-node-editor
    magic_enum::magic_enum
    httplib::httplib
    OpenSSL::SSL
    OpenSSL::Crypto
    nlohmann_json::nlohmann_json
)

if(MSVC)
    set_source_files_properties(${SHADERTOY_GUI_SOURCES} PROPERTIES
        COMPILE_OPTIONS "/Wall;/WX;/EHsc;/wd4514;/wd4820;/wd4623;/wd4625;/wd4626;/wd5026;/wd5027;/wd4061;/wd5045;/wd4365;/wd4868;/wd4710;/wd5262;/wd4711;/wd4738"
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_source_files_properties(${SHADERTOY_GUI_SOURCES} PROPERTIES
        COMPILE_OPTIONS "-Wall;-Wextra;-Werror;-Wconversion;-Wshadow=compatible-local;-Wno-psabi;-Wno-array-bounds"
    )
endif()

install(TARGETS shadertoy RUNTIME DESTINATION . BUNDLE DESTINATION .)
