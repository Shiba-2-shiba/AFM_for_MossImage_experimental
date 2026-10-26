# Final self-review

Scope: afm/, node entry point/schema, example API differences, runtime integration.
The native review surface failed with model_not_found; this is an author review.

- Image-first partition and full-grid FFT were compared with an independent
  dense joint-softmax oracle; chunked execution agrees within 2e-5 tolerance.
- Native baseline dispatch removes only AFM recursion state, preserves the
  actual block preference and consumes container inputs once.
- Non-selected image branches and all text output rows are kept from the
  same call's native baseline.
- Step identity is sampler callback-owned; real Euler and ER-SDE loops cover
  the adjusted first sigma and last step.
- Run options are restored before final logging; interrupts propagate.
- Workspace estimate includes retained QKV, baseline/previous/candidate
  outputs, FP32 scratch and complex FFT buffers. It is a budget estimate,
  not a total-device-memory guarantee.
- Detailed quality analysis and performance benchmarking are outside this
  implementation slice.
- Installation, server registration and browser round-trip are unverified
  because the target directory is outside the write scope.
- UI export is a draft; API examples are the verified executable format.

No Critical/Important code finding remains from this pass.
Tooling gaps: no installed ruff/typechecker, no independent reviewer.
