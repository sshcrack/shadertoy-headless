#include <shadertoy/shadertoy.h>

#include <stdio.h>

int main(void) {
    st_project* project = st_project_create("c-api-smoke");
    if(project == NULL) {
        const char* error = st_last_error();
        fprintf(stderr, "%s\n", error != NULL ? error : "unknown C API error");
        return 1;
    }

    st_project_destroy(project);
    puts("shadertoy-c-smoke=OK");
    return 0;
}
