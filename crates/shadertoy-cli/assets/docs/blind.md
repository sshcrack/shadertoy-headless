Blind comparisons let an agent make a visual judgment before learning which
source or parameter values produced each anonymous variant.

Create a blind comparison from arbitrary existing images or render directories:

  shadertoy blind create old.png new.png
  shadertoy blind create target/old-renders target/new-renders \
    --output-dir target/old-vs-new

Image directories are searched recursively for PNG/JPEG files and sorted
deterministically. Every source must contain the same number of images at the
same dimensions. The contact sheet uses anonymous variants A/B/C as columns and
corresponding source images as rows.

Project directories and built STTF artifacts can be rendered at matching frames:

  shadertoy blind create old-project new-project \
    --frames 0,60,120 --width 1280 --height 720

  shadertoy blind create old.sttf new.sttf \
    --frames 0,60,120 --width 1280 --height 720

Project sources use their manifest render defaults unless width/height/fps are
overridden. STTF sources default to 1280x720 at 60 fps unless overridden.

Git revisions can be compared without manually checking them out:

  shadertoy blind create \
    'git:v2.2.6::examples/demo' \
    'git:HEAD::examples/demo' \
    --frames 0,60,120

Use --git-root PATH when the command is not running inside the repository. The
form git:REF points at the worktree root; git:REF::SUBDIR selects a file or
directory inside that revision. Comparing a live working tree against its
committed base is therefore as simple as passing the project directory as one
source and git:HEAD::PATH as the other.

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
