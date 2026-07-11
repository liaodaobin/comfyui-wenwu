# Changelog

## 2026-07-11

- Clarified that `comfyui-wenwu` is a multi-workflow node pack, not only a Bernini prompt node.
- Documented the three model-facing functions separately:
  - `Bernini Prompt wenwu` for Bernini image and video prompt workflows.
  - `WenWu Krea2 Prompt Instruct` for Krea2 text-to-image and style-reference prompt rewriting.
  - `ACE Step 1.5 LLM WENWU` for ACE Step 1.5 music planning and conditioning output.
- Added a model-focused workflow table to the README so users can quickly pick the right node for their target model.
- Noted that Krea2 depends on `ComfyUI-llama-cpp`, while ACE Step 1.5 requires a separate ACE Step 1.5 ComfyUI setup and model files.
- Improved optional dependency handling so missing Krea2 or ACE Step-specific dependencies do not block unrelated WenWu nodes from loading.
