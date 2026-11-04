# K Row Balance range and patcher compatibility plan

**User request:** Increase strength and maximum change, select blocks such as
0,1,2,22,24,26,30,31,40,41, and fix the reported standard-ModelPatcher exception.

**Evidence:** The traceback stops at the exact-type/dynamic guard before reading or
editing weights. It establishes a rejected patcher, but does not reveal whether it
is a normal subclass or Dynamic. Old default strength0 bypassed that guard.
The local/public initial version limits cap to0.10; the report of cap0.3 succeeding
may be a different value or version and is not treated as evidence of a math fault.

**Design:** Preserve node ID, first three inputs, defaults0/.03 and legacy block41.
Append optional String `blocks="41"`, reuse the existing comma/range/all parser,
and allow valid block indices0–41. The user's dotted list is interpreted as integer
blocks30,31,40. Apply common strength/cap to selected blocks, preserving Q/V and codes.
Accept normal ModelPatcher subclasses; for Dynamic use native cached
`get_non_dynamic_delegate()` before normal object replacement. Check conflicts before
and after delegation, and never mutate Dynamic internals directly. Missing native
delegate/reload support must produce a precise compatibility error.

**Bounds:** strength[-10,10], cap[0,.95]. Formula/defaults unchanged. Positive strength
above1 can overshoot the median; no promise of monotonic normalization or quality.
Report clipped rows/fraction because raising strength cannot exceed the cap.

**Atomicity:** Validate every target first. Build all replacements before attaching
one cloned patcher, with one fresh cache UUID and marker. Failure on any block must
leave all input-model modules unchanged. No stacking ambiguous K patches.

**Validation:** TDD for normal-subclass regression, native-delegate API behavior in
a CPU harness (not GPU Dynamic execution), absent loader, multi-block apply/restore,
conflicts outside block41, input immutability and legacy schema. Test stronger math
and all requested real K tensors with bounded CPU probes. Keep checkpoint and
existing image/workflow inputs untouched. No generation, dependency install or driver change.

- [x] Math executor: bounds/constants/saturation metrics and unit tests.
- [x] Root: multi-block adapter, cached delegate boundary, V3 appended input and tests.
- [x] Tool executor: configurable example/probe arguments and tests, no broad grid.
- [x] Root: docs, independent review, full suites and real CPU checks.
- [ ] Publish the tested follow-up through the already-authorized GitHub repository
  workflow if accessible; preserve local Git limitations and unrelated user files.

Validation:133 tests/377 subtests with CPU native contracts, analysis172 tests, real10-block projection controls and unchangedcheckpoint. GPU Dynamic execution and imagequality unverified.
