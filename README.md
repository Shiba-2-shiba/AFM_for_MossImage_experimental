# MOSS Image AFM for ComfyUI

Experimental Attention Frequency Modulation for native MOSS-Image.
The node edits image-query/text-key logits on the actual latent grid, followed by
joint normalization over image and text keys. The model weights are unchanged.

## Installation

Place this package's contents in `ComfyUI/custom_nodes/AFM_for_MossImage` and restart
ComfyUI using the existing launch configuration. No additional runtime dependencies.
The current staged source directory is `参考/implementation/AFM_for_MossImage`;
deployment into the intended parent directory is tracked separately.

Use **MOSS Image AFM** between Load Diffusion Model and KSampler.
Start with `mode=observe`, then `mode=edit`, strength 0.05, block_scope 0.
These settings are experimental starting points, not validated quality recommendations.

The supplied API workflows retain the user's model and 1024×1024, 25-step,
CFG 2.5, ER-SDE/simple generation settings. API JSON is submitted through the API;
it is not a UI graph export.

## Controls

- `off` and edit strength 0 return a clone without AFM hooks.
- `observe` records selected joint blocks without FFT or manual attention.
- `edit` filters only the selected CFG image rows. Text rows and unselected
  branch rows preserve the same call's native output.
- `block_scope` accepts indices/ranges or all; text refinement blocks are excluded.
- `start_percent/end_percent` use the actual executed sampler step schedule.
- `max_workspace_mib` limits a conservative AFM workspace estimate, not total VRAM.
- `fail_mode=raise` stops on edit failure. `fallback_original` returns that call's
  complete native output and records the failure.
- Summary logs are in `output/moss_afm/<run_id>.jsonl`. Detailed QKV are not saved.

## Initial scope

Native ComfyUI MOSS only; standard Euler and ER-SDE.
Fused/unfused QKV, rectangular image grids, CFG batching and separate text lengths
are covered by the test plan. HF plugin loaders, torch.compile, multi-GPU,
skip-cache, other block/attention patches and Slider interoperability are not qualified.

See [implementation status](docs/implementation-status.md) for verified conditions.
Quality evaluation is a separate project; image differences do not establish improvement.

## Tests

```powershell
python -m pytest -q
$env:MOSS_COMFY_ROOT='C:/Users/inott/ComfyUI-Installs/ComfyUI/ComfyUI'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/ComfyUI/.venv/Scripts/python.exe' -m unittest discover -s tests -t . -v
```

CPU-only tests can run with an existing PyTorch/pytest installation.
Actual ComfyUI contract tests must use the ComfyUI Python environment.
The GPU smoke helper uses a separate process and writes only to its specified output.
