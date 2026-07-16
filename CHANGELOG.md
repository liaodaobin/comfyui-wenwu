# Changelog

## 2026-07-16

- Added the independent `WenWu LTX-2.3 Prompt Enhancer` without replacing or rewriting the Bernini, Krea2, or ACE Step prompt nodes.
- Added up to five reference-image inputs: References 1-4 map to Licon subject slots and Reference 5 maps to the dedicated background slot.
- Added subject-identity separation, incidental portrait-background exclusion, original-language dialogue preservation, duration-aware story pacing, camera direction, synchronized sound, and optional music guidance.
- Added configurable duration, frame rate, resolution, seed, and LTX-compatible frame-count outputs.
- Added a locked/editable enhanced-prompt panel and a prompt-only refresh action. Locked execution regenerates through Llama; edited execution uses the confirmed prompt without another Llama pass.
- Added llama.cpp/mmproj cleanup before downstream video inference.
- Fixed multi-reference visual preprocessing so separately connected single images are actually resized for Llama analysis. Subject slots use a 640-pixel longest edge and the dedicated background slot uses 768 pixels, while original workflow images remain untouched for Licon MSR and LTX-2.3.
- Added optional import handling for the LTX-2.3 node so an unavailable model-specific dependency does not prevent unrelated WenWu nodes from loading.

## 2026-07-11

- Clarified that `comfyui-wenwu` is a multi-workflow node pack, not only a Bernini prompt node.
- Documented the three model-facing functions separately:
  - `Bernini Prompt wenwu` for Bernini image and video prompt workflows.
  - `WenWu Krea2 Prompt Instruct` for Krea2 text-to-image and style-reference prompt rewriting.
  - `ACE Step 1.5 LLM WENWU` for ACE Step 1.5 music planning and conditioning output.
- Added a model-focused workflow table to the README so users can quickly pick the right node for their target model.
- Noted that Krea2 depends on `ComfyUI-llama-cpp`, while ACE Step 1.5 requires a separate ACE Step 1.5 ComfyUI setup and model files.
- Improved optional dependency handling so missing Krea2 or ACE Step-specific dependencies do not block unrelated WenWu nodes from loading.
