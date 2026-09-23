Blind comparisons let an agent make a visual judgment before learning which
source or parameter values produced each anonymous variant.

Create a blind comparison from arbitrary existing images or render directories:

  shadertoy blind create old.png new.png
  shadertoy blind create target/old-renders target/new-renders \
    --output-dir target/old-vs-new

Source auto-detection prefers a ShaderToy project, then a unique STTF build,
then PNG/JPEG images. This keeps project texture assets from being mistaken for
comparison inputs. Image-only directories are searched recursively and sorted
deterministically. Every source must contain the same number of images at the
same dimensions unless every source uses the explicit `project:` form below.
The contact sheet uses anonymous variants A/B/C as columns and corresponding
source images as rows.

Project directories and built STTF artifacts can be rendered at matching frames:

  shadertoy blind create old-project new-project \
    --frames 0,60,120 --width 1280 --height 720

  shadertoy blind create old.sttf new.sttf \
    --frames 0,60,120 --width 1280 --height 720

Project sources use their manifest render defaults unless width/height/fps are
overridden. STTF sources default to 1280x720 at 60 fps unless overridden.

Named quality presets can be compared directly without temporary project copies:

  shadertoy blind create \
    'project:.@preset=high' \
    'project:.@preset=medium' \
    'project:.@preset=low' \
    --frames 60,180,300

When all inputs use explicit `project:` sources, differing preset output scales
are normalized to the largest source resolution for the blinded contact sheet
and variant images. The sources must keep the same aspect ratio. This
normalization is intentionally not applied to ordinary image/path comparisons.

Git revisions can be compared without manually checking them out:

  shadertoy blind create \
    'git:v2.2.6::examples/demo' \
    'git:HEAD::examples/demo' \
    --frames 0,60,120

Use --git-root PATH when the command is not running inside the repository. When
the command runs inside a nested ShaderToy project, bare `git:REF` sources keep
that project-relative path across revisions. Otherwise a bare revision is
auto-detected from the worktree root; a single nested ShaderToy project or STTF
build is preferred over image assets. `git:REF::SUBDIR` always selects an
explicit file or directory and overrides implicit project selection. Comparing
a live working tree against its committed base can therefore use the project
directory as one source and either `git:HEAD` from inside that project or
`git:HEAD::PATH` explicitly as the other.

Parameter sweeps can still create the same blind workflow directly:

  shadertoy sweep --blind --frame 120 --set u_foam_gain=0.8,1.0,1.2

Both forms randomize the variant order, create blind-contact-sheet.png, and
write anonymous A/B/C outputs. The public blind-session.json and normal command
output intentionally omit the source/parameter mapping.

Record the visual decision before revealing:

  shadertoy blind judge target/blind-comparison/blind-session.json \
    --pick B \
    --reason "Best crest breakup without flattening the mid-frequency chop."

For longer notes, use --reason-file notes.md instead of --reason. A session
accepts one judgment and cannot be judged after it has been revealed.

Then reveal:

  shadertoy blind reveal target/blind-comparison/blind-session.json

Reveal is refused until a judgment exists. It writes blind-reveal.json with the
recorded judgment, selected variant identity/settings, and the full A/B/C
mapping. JSON mode returns the same combined report for agent tooling.

The hidden .blind-mapping.bin artifact is intentionally opaque to ordinary text
inspection so identities do not leak into normal agent context before judgment.
This is a workflow guard against evaluation bias, not a security or
cryptographic boundary: an actor deliberately inspecting implementation details
can bypass it.

Shared custom-uniform overrides can be applied to every rendered source with repeatable `--set NAME=VALUE`, e.g. `shadertoy blind create old-project new-project --set u_storm=1`. Static image sources are unaffected.
