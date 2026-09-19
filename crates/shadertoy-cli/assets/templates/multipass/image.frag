void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;

    // The manifest wires current-frame Buffer A to Image iChannel0.
    vec3 buffered = texture(iChannel0, uv).rgb;
    vec3 background = vec3(0.015, 0.02, 0.035);

    fragColor = vec4(background + buffered, 1.0);
}
