/*
    SPDX-License-Identifier: Apache-2.0
    Copyright 2023-2026 Yingwei Zheng and contributors
*/
#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(SHADERTOY_C_STATIC)
#define ST_NATIVE_EXPORT
#elif defined(SHADERTOY_C_BUILD)
#define ST_NATIVE_EXPORT __declspec(dllexport)
#else
#define ST_NATIVE_EXPORT __declspec(dllimport)
#endif
#else
#define ST_NATIVE_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct st_context st_context;
typedef struct st_project st_project;
typedef struct st_runtime st_runtime;

typedef uint32_t st_pass_kind;
/* Plain constants (not an enum): MSVC fixes enum type to int while GCC/Clang
 * pick unsigned, which made bindgen emit i32 on Windows but u32 elsewhere. */
static const st_pass_kind ST_PASS_IMAGE = 0;
static const st_pass_kind ST_PASS_BUFFER = 1;
static const st_pass_kind ST_PASS_CUBEMAP = 2;

typedef uint32_t st_input_kind;
static const st_input_kind ST_INPUT_PASS = 0;
static const st_input_kind ST_INPUT_TEXTURE = 1;
static const st_input_kind ST_INPUT_KEYBOARD = 2;
static const st_input_kind ST_INPUT_MUSIC = 3;
static const st_input_kind ST_INPUT_CUBEMAP = 4;
static const st_input_kind ST_INPUT_VOLUME = 5;

typedef uint32_t st_filter;
static const st_filter ST_FILTER_MIPMAP = 0;
static const st_filter ST_FILTER_LINEAR = 1;
static const st_filter ST_FILTER_NEAREST = 2;

typedef uint32_t st_wrap;
static const st_wrap ST_WRAP_CLAMP = 0;
static const st_wrap ST_WRAP_REPEAT = 1;

ST_NATIVE_EXPORT const char* st_last_error(void);

/* Offscreen context lifecycle must be created and destroyed on the process main thread.
 * Linux uses EGL's surfaceless platform directly and requires no X11/Wayland display. */
ST_NATIVE_EXPORT st_context* st_context_create_hidden(uint32_t width, uint32_t height);
ST_NATIVE_EXPORT int st_context_make_current(st_context* context);
ST_NATIVE_EXPORT void st_context_destroy(st_context* context);

ST_NATIVE_EXPORT st_project* st_project_create(const char* name);
ST_NATIVE_EXPORT void st_project_destroy(st_project* project);
ST_NATIVE_EXPORT int st_project_add_pass(st_project* project, const char* name, st_pass_kind kind, const char* source);
ST_NATIVE_EXPORT int st_project_add_input(st_project* project, const char* pass_name, uint32_t channel, st_input_kind kind,
                                          const char* source, int previous_frame, st_filter filter, st_wrap wrap);
ST_NATIVE_EXPORT int st_project_add_texture_rgba8(st_project* project, const char* name, uint32_t width, uint32_t height,
                                                  const uint8_t* rgba, size_t rgba_len);
ST_NATIVE_EXPORT int st_project_add_cubemap_rgba8(st_project* project, const char* name, uint32_t size, const uint8_t* rgba,
                                                  size_t rgba_len);
ST_NATIVE_EXPORT int st_project_add_volume_u8(st_project* project, const char* name, uint32_t size, uint32_t channels,
                                              const uint8_t* data, size_t data_len);

ST_NATIVE_EXPORT st_runtime* st_runtime_create(void);
ST_NATIVE_EXPORT void st_runtime_destroy(st_runtime* runtime);
ST_NATIVE_EXPORT int st_runtime_load_project(st_runtime* runtime, const st_project* project);
ST_NATIVE_EXPORT int st_runtime_save_sttf(const st_runtime* runtime, const char* path);
ST_NATIVE_EXPORT void st_runtime_tick(st_runtime* runtime, float frame_rate);
ST_NATIVE_EXPORT void st_runtime_tick_fixed(st_runtime* runtime, float delta_seconds, float frame_rate);
ST_NATIVE_EXPORT void st_runtime_reset_time(st_runtime* runtime);
ST_NATIVE_EXPORT void st_runtime_set_fixed_state(st_runtime* runtime, float time_seconds, int32_t frame, float frame_rate);
ST_NATIVE_EXPORT float st_runtime_time(const st_runtime* runtime);
ST_NATIVE_EXPORT int32_t st_runtime_frame(const st_runtime* runtime);
ST_NATIVE_EXPORT void st_runtime_pause(st_runtime* runtime);
ST_NATIVE_EXPORT void st_runtime_resume(st_runtime* runtime);
ST_NATIVE_EXPORT int st_runtime_is_running(const st_runtime* runtime);
ST_NATIVE_EXPORT float st_runtime_time_scale(const st_runtime* runtime);
ST_NATIVE_EXPORT void st_runtime_set_time_scale(st_runtime* runtime, float log2_scale);
ST_NATIVE_EXPORT int st_runtime_set_mouse(st_runtime* runtime, float x, float y, int down, int clicked);
ST_NATIVE_EXPORT int st_runtime_clear_mouse(st_runtime* runtime);
ST_NATIVE_EXPORT int st_runtime_set_key(st_runtime* runtime, uint8_t key, int down, int pressed);
ST_NATIVE_EXPORT int st_runtime_clear_key_transients(st_runtime* runtime);

ST_NATIVE_EXPORT int st_runtime_render_rgb(st_runtime* runtime, uint32_t width, uint32_t height, uint8_t* out_rgb,
                                           size_t out_len);
ST_NATIVE_EXPORT int st_runtime_snapshot_pass_rgb(st_runtime* runtime, const char* pass_name, uint8_t* out_rgb, size_t out_len);
ST_NATIVE_EXPORT int st_runtime_snapshot_pass_rgba32f(st_runtime* runtime, const char* pass_name, float* out_rgba,
                                                      size_t out_len);
ST_NATIVE_EXPORT int st_runtime_override_pass_rgba8(st_runtime* runtime, const char* pass_name, uint32_t width, uint32_t height,
                                                    const uint8_t* rgba, size_t rgba_len);
ST_NATIVE_EXPORT int st_runtime_restore_pass_rgba32f(st_runtime* runtime, const char* pass_name, uint32_t width, uint32_t height,
                                                     const float* rgba, size_t rgba_len);

#ifdef __cplusplus
}
#endif
