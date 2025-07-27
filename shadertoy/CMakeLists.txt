file(GLOB SHADERTOY_SRC "*.cpp" "NodeEditor/*.cpp")
file(GLOB SHADERTOY_HEADERS "*.hpp" "NodeEditor/*.hpp")

option(ENABLE_DEBUG_WINDOW "Enable debug window" OFF)

find_package(fmt CONFIG REQUIRED)
find_package(GLEW CONFIG REQUIRED)
find_package(Microsoft.GSL CONFIG REQUIRED)
find_package(magic_enum CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(httplib CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)
find_path(CPP_BASE64_INCLUDE_DIRS "cpp-base64/base64.cpp")

if(ENABLE_DEBUG_WINDOW)
    message(STATUS "Enabling debug window for Shadertoy headless")
    add_subdirectory(thirdparty)

    hello_imgui_add_app(shadertoy-headless
        ${SHADERTOY_SRC}
        ${CPP_BASE64_INCLUDE_DIRS}/cpp-base64/base64.cpp
    )
else()
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

    message(STATUS "Disabling debug window for Shadertoy headless Using sources from ${SHADERTOY_SRC}")
    add_library(shadertoy-headless
        ${SHADERTOY_SRC}
        ${SHADERTOY_HEADERS}
        ${CPP_BASE64_INCLUDE_DIRS}/cpp-base64/base64.cpp
    )
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set_source_files_properties(${SHADERTOY_SRC} PROPERTIES COMPILE_FLAGS "/Wall /WX /wd4514 /wd4820 /wd4623 /wd4625 /wd4626 /wd5026 /wd5027 /wd4061 /wd5045 /wd4365 /wd4868 /wd4710 /wd5262 /wd4711 /wd4738 /D_CRT_SECURE_NO_WARNINGS /D_SILENCE_ALL_MS_EXT_DEPRECATION_WARNINGS")
elseif(CMAKE_COMPILER_IS_GNUCXX)
	set_source_files_properties(${SHADERTOY_SRC} PROPERTIES COMPILE_FLAGS "-Wall -Wextra -Werror -Wconversion -Wshadow=compatible-local -Wno-psabi -Wno-array-bounds -Wno-unused-parameter")
endif()

if(NOT ENABLE_DEBUG_WINDOW)
    find_package(nlohmann_json 3.2.0 REQUIRED)
    find_package(hello-imgui CONFIG REQUIRED)
    target_link_libraries(shadertoy-headless PRIVATE hello-imgui::hello_imgui nlohmann_json::nlohmann_json)
endif()

#target_include_directories(shadertoy-headless PRIVATE ${CMAKE_CURRENT_LIST_DIR}/thirdparty/hello_imgui/src ${IMGUI_SRC_DIR} ${CMAKE_CURRENT_LIST_DIR}/thirdparty/)
target_link_libraries(shadertoy-headless PRIVATE fmt::fmt GLEW::GLEW Microsoft.GSL::GSL magic_enum::magic_enum httplib::httplib OpenSSL::SSL OpenSSL::Crypto)
target_include_directories(shadertoy-headless PRIVATE ${Stb_INCLUDE_DIR} ${CPP_BASE64_INCLUDE_DIRS} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_features(shadertoy-headless PRIVATE cxx_std_23)
set_target_properties(shadertoy-headless PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(APPLE)
    find_library(AppleSecurityLib Security)
    target_link_libraries(shadertoy-headless PRIVATE ${AppleSecurityLib})
    #   set_target_properties(shadertoy-headless PROPERTIES
    #	    MACOSX_BUNDLE TRUE
    #		MACOSX_BUNDLE_BUNDLE_NAME "Shadertoy live viewer"
    #		MACOSX_BUNDLE_BUNDLE_VERSION ${CMAKE_PROJECT_VERSION}
    #		MACOSX_BUNDLE_SHORT_VERSION_STRING ${CMAKE_PROJECT_VERSION}
    #		MACOSX_BUNDLE_COPYRIGHT "Copyright 2023-2025 Yingwei Zheng"
    #		MACOSX_BUNDLE_GUI_IDENTIFIER "com.dtcxzyw.shadertoy"
    #       MACOSX_FRAMEWORK_IDENTIFIER "com.dtcxzyw.shadertoy"
    #	)
endif(APPLE)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE DESTINATION .)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../licenses DESTINATION .)
#install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../examples DESTINATION .)
install(TARGETS shadertoy-headless DESTINATION .)
#install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp DESTINATION shadertoy)
#install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/NodeEditor/*.hpp DESTINATION shadertoy/NodeEditor)

# if(APPLE)
# 	include(InstallRequiredSystemLibraries)
# 	set(CPACK_GENERATOR "DRAGNDROP")
#     set(CPACK_RESOURCE_FILE_LICENSE ${CMAKE_SOURCE_DIR}/LICENSE)
# 	set(CPACK_PACKAGE_VENDOR "Yingwei Zheng")
# 	include(CPack)
# endif(APPLE)
