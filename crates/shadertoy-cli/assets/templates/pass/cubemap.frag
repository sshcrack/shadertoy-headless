void mainCubemap(out vec4 fragColor, in vec2 fragCoord, in vec3 rayOrigin, in vec3 rayDirection)
{
    vec3 direction = normalize(rayDirection);
    fragColor = vec4(direction * 0.5 + 0.5, 1.0);
}
