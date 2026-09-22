Graph validation
================

Use the top-level graph command for a resolved view of pass inputs, feedback,
SSBO sharing, dimensions, and advisory diagnostics:

  shadertoy graph --json
  shadertoy graph --preset low --dot target/graph.dot

Graphviz DOT output includes pass, asset, and SSBO nodes. Previous-frame edges
are dashed.

shadertoy check always performs hard graph validation and real GLSL compilation.
Add --pedantic to enable advisory resource/architecture checks and make warnings
fail the command:

  shadertoy check --pedantic

Current pedantic diagnostics include:
- unreachable non-Sound passes;
- declared but unused assets;
- declared custom uniforms not referenced by expanded shader source;
- feedback buffers that inherit viewport resolution;
- shared SSBO users that have no current-frame dependency ordering them.

A current-frame dependency cycle is always an error. Previous-frame feedback is
not treated as a current-frame cycle.
