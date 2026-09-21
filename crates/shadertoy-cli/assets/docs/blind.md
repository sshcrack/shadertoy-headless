Blind comparisons let an agent judge rendered variants before learning which
parameter values produced them.

Create a blinded sweep:

  shadertoy sweep --blind --frame 120 --set u_foam_gain=0.8,1.0,1.2

The sweep randomizes the variant order, writes anonymous images such as A.png,
B.png, and C.png, and creates blind-contact-sheet.png. Contact-sheet cells are
row-major in label order: A, B, C, and so on. The public blind-session.json and
normal command output intentionally omit the parameter mapping.

Record the visual decision before revealing:

  shadertoy blind judge target/sweep/blind-session.json \
    --pick B \
    --reason "Best crest breakup without flattening the mid-frequency chop."

For longer notes, use --reason-file notes.md instead of --reason. A session
accepts one judgment and cannot be judged after it has been revealed.

Then reveal:

  shadertoy blind reveal target/sweep/blind-session.json

Reveal is refused until a judgment exists. It writes blind-reveal.json containing
the recorded judgment, the selected variant's real parameter assignments, and the
full A/B/C-to-parameter mapping. JSON mode returns the same combined report for
agent tooling.

The hidden .blind-mapping.bin artifact is intentionally opaque to ordinary text
inspection so parameter values do not leak into normal agent context before the
judgment. This is a workflow guard against evaluation bias, not a security or
cryptographic boundary: an actor deliberately inspecting implementation details
can bypass it.
