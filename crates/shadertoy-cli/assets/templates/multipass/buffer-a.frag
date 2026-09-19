void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;

    // Buffer A receives its own previous frame through iChannel0.
    // Avoid reading undefined startup contents on the first frame.
    vec3 history = iFrame == 0
        ? vec3(0.0)
        : texture(iChannel0, uv).rgb * 0.985;

    vec2 p = uv - 0.5;
    float pulse = exp(-55.0 * dot(p, p)) * (0.55 + 0.45 * sin(iTime * 2.0));
    vec3 injected = pulse * vec3(0.2, 0.6, 1.0);

    fragColor = vec4(max(history, injected), 1.0);
}
