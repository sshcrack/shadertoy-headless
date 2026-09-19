#include <shadertoy/shadertoy.h>

#include <stdint.h>
#include <stdio.h>

int main(void) {
    if(st_context_create_hidden(UINT32_MAX, 1) != NULL) {
        fputs("oversized context dimensions were unexpectedly accepted\n", stderr);
        return 3;
    }

    st_runtime* runtime = st_runtime_create();
    if(runtime == NULL) {
        fputs("failed to create runtime for validation smoke\n", stderr);
        return 4;
    }
    uint8_t rgb[3] = { 0, 0, 0 };
    if(st_runtime_render_rgb(runtime, 0, 1, rgb, 0) == 0) {
        fputs("zero render dimensions were unexpectedly accepted\n", stderr);
        st_runtime_destroy(runtime);
        return 5;
    }
    st_runtime_destroy(runtime);

    st_project* project = st_project_create("c-api-smoke");
    if(project == NULL) {
        const char* error = st_last_error();
        fprintf(stderr, "%s\n", error != NULL ? error : "unknown C API error");
        return 1;
    }

    const uint8_t byte = 0;
    if(st_project_add_texture_rgba8(project, "overflow", UINT32_MAX, UINT32_MAX, &byte, 1) == 0) {
        fputs("overflowing texture dimensions were unexpectedly accepted\n", stderr);
        st_project_destroy(project);
        return 2;
    }

    st_project_destroy(project);
    puts("shadertoy-c-smoke=OK");
    return 0;
}
