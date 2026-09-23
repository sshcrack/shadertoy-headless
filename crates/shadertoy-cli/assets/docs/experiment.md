Experiments
===========

Use shadertoy experiment when comparing complete shader states rather than one
parameter sweep. Sources use the same detection rules as blind create: images,
image directories, ShaderToy projects, STTF builds, git revisions, and explicit
quality presets such as project:.@preset=medium.

A normal baseline/candidate comparison:

  shadertoy experiment \
    --baseline git:main \
    --candidate git:HEAD \
    --frames 0,60,120 \
    --metric rmse,ssim \
    --output-dir target/experiment

Use repeatable `--set NAME=VALUE` overrides to render and profile every project/STTF source in the same custom-uniform state. For example, a storm comparison can use `--set u_storm=1`; static image sources are unaffected.

Repeat --candidate or --variant for N-way experiments. Project and git sources
are profiled by default; use --profile-samples 0 when only visual metrics are
wanted. The output directory contains deterministic variant renders, a contact
sheet, and experiment-report.json with per-frame and aggregate metrics.

Bias-resistant visual review
----------------------------

Add --blind to randomize sources behind anonymous labels and create the same
sealed judgment lifecycle used by blind create:

  shadertoy experiment \
    --baseline 'project:.@preset=high' \
    --candidate 'project:.@preset=medium' \
    --candidate 'project:.@preset=low' \
    --frames 60,180,300 \
    --blind

Inspect the anonymous contact sheet first, then record the visual rationale:

  shadertoy blind judge target/experiment/blind-session.json \
    --pick B --reason "cleanest temporal breakup"

Only then reveal the source mapping:

  shadertoy blind reveal target/experiment/blind-session.json

Supported metrics are normalized RGB RMSE and windowed luminance SSIM. The
report also records available git/manifest provenance and GPU profiling data so
the experiment can be reproduced.
