file(GLOB SHADERTOY_SRC "*.cpp" "NodeEditor/*.cpp")

find_package(fmt CONFIG REQUIRED)
find_package(GLEW CONFIG REQUIRED)
find_package(Microsoft.GSL CONFIG REQUIRED)
find_package(magic_enum CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(httplib CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)
find_path(CPP_BASE64_INCLUDE_DIRS "cpp-base64/base64.cpp")

add_subdirectory(thirdparty)

#hello_imgui_add_app(shadertoyLib
#	${SHADERTOY_SRC}
#	${CPP_BASE64_INCLUDE_DIRS}/cpp-base64/base64.cpp
#)

add_library(shadertoyLib
		${SHADERTOY_SRC}
		${CPP_BASE64_INCLUDE_DIRS}/cpp-base64/base64.cpp
)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	set_source_files_properties(${SHADERTOY_SRC} PROPERTIES COMPILE_FLAGS "/Wall /WX /wd4514 /wd4820 /wd4623 /wd4625 /wd4626 /wd5026 /wd5027 /wd4061 /wd5045 /wd4365 /wd4868 /wd4710 /wd5262 /wd4711 /wd4738 /D_CRT_SECURE_NO_WARNINGS /D_SILENCE_ALL_MS_EXT_DEPRECATION_WARNINGS")
elseif(CMAKE_COMPILER_IS_GNUCXX)
	set_source_files_properties(${SHADERTOY_SRC} PROPERTIES COMPILE_FLAGS "-Wall -Wextra -Werror -Wconversion -Wshadow -Wno-psabi -Wno-array-bounds -Wno-unused-parameter")
endif()
target_include_directories(shadertoyLib PRIVATE ${CMAKE_CURRENT_LIST_DIR}/thirdparty/hello_imgui/src ${IMGUI_SRC_DIR} ${CMAKE_CURRENT_LIST_DIR}/thirdparty/)
target_link_libraries(shadertoyLib PRIVATE fmt::fmt GLEW::GLEW Microsoft.GSL::GSL magic_enum::magic_enum httplib::httplib OpenSSL::SSL OpenSSL::Crypto)
target_include_directories(shadertoyLib PRIVATE ${Stb_INCLUDE_DIR} ${CPP_BASE64_INCLUDE_DIRS})

if(APPLE)
	find_library(AppleSecurityLib Security)
	target_link_libraries(shadertoyLib PRIVATE ${AppleSecurityLib})
#   set_target_properties(shadertoyLib PROPERTIES
#	    MACOSX_BUNDLE TRUE
#		MACOSX_BUNDLE_BUNDLE_NAME "Shadertoy live viewer"
#		MACOSX_BUNDLE_BUNDLE_VERSION ${CMAKE_PROJECT_VERSION}
#		MACOSX_BUNDLE_SHORT_VERSION_STRING ${CMAKE_PROJECT_VERSION}
#		MACOSX_BUNDLE_COPYRIGHT "Copyright 2023-2025 Yingwei Zheng"
#		MACOSX_BUNDLE_GUI_IDENTIFIER "com.dtcxzyw.shadertoy"
#       MACOSX_FRAMEWORK_IDENTIFIER "com.dtcxzyw.shadertoy"
#	)
endif(APPLE)

install(FILES ${CMAKE_SOURCE_DIR}/LICENSE DESTINATION .)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/licenses DESTINATION .)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/examples DESTINATION .)
install(TARGETS shadertoyLib DESTINATION .)

# if(APPLE)
# 	include(InstallRequiredSystemLibraries)
# 	set(CPACK_GENERATOR "DRAGNDROP")
#     set(CPACK_RESOURCE_FILE_LICENSE ${CMAKE_SOURCE_DIR}/LICENSE)
# 	set(CPACK_PACKAGE_VENDOR "Yingwei Zheng")
# 	include(CPack)
# endif(APPLE)
