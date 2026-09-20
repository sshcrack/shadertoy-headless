vec2 mainSound(int samp, float time) {
    float frequency = 220.0;
    float wave = sin(6.28318530718 * frequency * time) * 0.2;
    return vec2(wave);
}
