find_package(glfw3 CONFIG REQUIRED)

add_library(shadertoy-c SHARED src/shadertoy.cpp)
add_library(shadertoy::c ALIAS shadertoy-c)
set_target_properties(shadertoy-c PROPERTIES EXPORT_NAME c)
target_compile_features(shadertoy-c PRIVATE cxx_std_23)
target_compile_definitions(shadertoy-c PRIVATE SHADERTOY_C_BUILD)
target_link_libraries(shadertoy-c PRIVATE shadertoy::shadertoy glfw)
target_include_directories(shadertoy-c PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
set_target_properties(shadertoy-c PROPERTIES
    OUTPUT_NAME shadertoy_c
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

if(MSVC)
    target_compile_options(shadertoy-c PRIVATE /Wall /WX /wd4514 /wd4820 /wd4625 /wd4626 /wd5026 /wd5027)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(shadertoy-c PRIVATE -Wall -Wextra -Werror -Wconversion -Wshadow=compatible-local)
endif()

install(TARGETS shadertoy-c
    EXPORT shadertoy-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(BUILD_TESTING)
    enable_language(C)
    add_executable(shadertoy-c-smoke tests/smoke.c)
    target_link_libraries(shadertoy-c-smoke PRIVATE shadertoy-c)
    add_test(NAME shadertoy-c-smoke COMMAND shadertoy-c-smoke)
endif()
