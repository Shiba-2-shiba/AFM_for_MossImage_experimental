# Experimental K row-balance node implementation plan

> Execute the user's authorized local implementation directly with TDD and a separate code review. No server, image generation, optimizer, installation, or checkpoint write.

**Goal:** Add a separate V3 node for testing whether gently reducing block41's within-head K-row norm dispersion helps image quality. Improvement is a hypothesis, not an acceptance claim.

**Architecture:** For native MOSS's canonical fused INT8 ConvRot QKV, modify only the K rows of the FP32 row-scale tensor. Keep every INT8 code and the Q/V scales unchanged. Build a fresh native mixed-precision Linear from its serialized state and install it through a cloned ModelPatcher's reversible object-patch lifecycle. Do not use ordinary LoRA/weight functions, which can dequantize/requantize the fused projection. Reject dynamic patchers and conflicting target patches initially. Invalidate the clone's weight-cache identity for each edited variant; zero strength is clone-only.

**Math:** For each 128-row head, r_i = scale_i * ||q_i||2, m = exact median(r). Gain is exp(clamp(strength*(log(m)-log(r_i)), log(1-cap), log(1+cap))). This uses input-rotation orthogonality; output-row scaling satisfies D(WH)=(DW)H. Positive strength compresses dispersion; negative strength is a reverse-direction control. Reject zero/nonfinite norms/scales. No cross-head equalization and no uniform boost to block0.

**Controls:** Fixed block41, strength default0 in [-1,1], advanced max_change default0.03 in [0,0.10]. Target shape [9216,3072], 24heads×128, 42blocks, bias-free QKV, int8_tensorwise/TensorWiseINT8Layout, ConvRot group256. Output MODEL plus a compact STRING report. No new dependencies or frontend JS.

**Files / ownership:** math executor owns `afm/k_balance_math.py` and `tests/unit/test_k_balance_math.py`; root owns `afm/k_balance.py`, node registration, CPU contract tests, docs, and examples. Existing untracked `examples/afm_validation_prompts_2026-10-28.txt` stays untouched. Work directly in the named repository as requested; no commits or other repo edits.

- [x] Lock math with failing tests: gain bounds/direction, identity, per-head scale invariance, zero/nonfinite rejection, no input mutation, chunk consistency.
- [x] Lock adapter with failing CPU-native tests: new module/state, only K scales, no requantization, original/clone isolation, cache invalidation, repeated application/restore, target-patch and unsupported-layout rejection.
- [x] Implement new V3 node and registration; verify real schema/execute with available ComfyUI core on CPU.
- [x] Add identity/compress/reverse UI examples from supplied original workflow, preserving actual VAE/prompt/seed/sampling. Examples are not executed; avoid combining Residual/AFM edits initially.
- [x] Run AFM's full suite including CPU native contracts; root analysis suite; syntax checks; independent review and material fixes.
- [x] Run a bounded CPU probe on only the real block41 QKV tensor, no full-model forward. Record raw-code/QV-scale equality, within-head dispersion change, and original checkpoint hash preservation.

**Validation limits:** A CPU small/native projection test is not GPU/driver/whole-image validation. The reference API/math can be checked with the installed Python3.10 environment, while the project's declared Python3.12+ deployment requirement remains unchanged. Native ComfyUI reference commit7f09d3f and installed comfy-kitchen0.2.39 are recorded; 0.2.40's relevant decoder math was previously cross-checked.

Final evidence: AFM119 tests/361 subtests + analysis172 tests pass; nativeCPU schema/patcher tests enabled; real-QKV zero/rebuild/positive/reverse probes recorded outside repository. Checkpoint unchanged, image generation not run.
