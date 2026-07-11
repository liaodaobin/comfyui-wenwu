# comfyui-wenwu

ComfyUI custom nodes for WenWu local prompt workflows, covering Bernini image/video prompts, Krea2 prompt rewriting, and ACE Step 1.5 music planning.

This node package is built for local generation inside ComfyUI. It lets users enter natural spoken-style descriptions and then uses local `llama.cpp` inference through `ComfyUI-llama-cpp` to convert them into structured prompts for different downstream tools.

The goal is to save users from manually writing strict prompt text while keeping the conversion local and reusable across visual and music workflows.

## What this node provides

- **Bernini Prompt wenwu**: converts natural user descriptions into professional Bernini-oriented prompt text for image and video tasks through a local `ComfyUI-llama-cpp` model.
- **WenWu Krea2 Prompt Instruct**: converts Chinese or mixed-language image ideas into a single polished English Krea2 prompt. It supports text-to-image prompt expansion and optional style-reference prompt rewriting.
- **ACE Step 1.5 LLM WENWU**: converts natural music descriptions into ACE Step 1.5-ready tags, lyrics, BPM, duration, language, key, and conditioning. It includes a preview/refresh UI and an editable confirmed prompt box.
- **Simple Text**: a lightweight canvas text node that can be edited by double-clicking the node or using the node context menu.

## Model-focused workflows

This package is not a single Bernini node. It contains three model-facing prompt helpers that share the same WenWu idea: write naturally first, then let a local LLM turn the request into the stricter format expected by the target model.

| Node | Target model or workflow | Best for | Main output |
| --- | --- | --- | --- |
| **Bernini Prompt wenwu** | Bernini image and video prompt workflows | Turning loose Chinese or mixed-language scene ideas into polished Bernini-style visual prompts | Structured prompt text for image/video generation |
| **WenWu Krea2 Prompt Instruct** | Krea2 text-to-image and style-reference workflows | Rewriting short image ideas into a single continuous English Krea2 prompt, with optional style-reference image guidance | Clean English Krea2 prompt text |
| **ACE Step 1.5 LLM WENWU** | ACE Step 1.5 music generation workflows | Planning songs from natural music descriptions, including style, lyrics, vocals, BPM, key, duration, and conditioning | ACE Step-ready tags, lyrics, prompt text, `conditioning`, and `duration` |

## Current status

This repository is synced from a local ComfyUI custom node directory:

```text
E:\ComfyUI-v3\ComfyUI\custom_nodes\comfyui-wenwu
```

Recent local changes replace the previous `WenWuShowAndSaveAnything` node export with `WenWuSimpleTextNode`. The old `WenWuShowAndSaveAnything` implementation is still present in `wenwu_prompt_generator.py`, but it is no longer registered in `nodes.py`.

`Simple Text` is implemented mostly in `js/showAnything.js`. It stores node text and style values in node properties and supports:

- multiline text
- font size, color, weight, and italic style
- left, center, and right alignment
- optional background color
- padding and border radius
- color swatches in the editor

`ACE Step 1.5 LLM WENWU` is implemented in `ace_step_llm_wenwu.py` with frontend controls in `js/ace_step_plan_preview.js`. It supports:

- natural-language music planning through a connected llama.cpp model loader
- preview/refresh prompt generation without running the full audio workflow
- editable confirmed prompt mode
- instrumental/no-vocal handling
- automatic TAGS line formatting
- style-based BPM, time signature, language, and key suggestions
- direct `conditioning` and `duration` outputs for ACE Step 1.5 workflows

`WenWu Krea2 Prompt Instruct` is implemented in `krea2_prompt_instruct.py` with frontend controls in `js/krea2_prompt_instruct.js`. It supports:

- local Krea2 prompt generation through a connected llama.cpp model loader
- Chinese or mixed-language input rewritten into one continuous English prompt
- optional style-reference mode with an uploaded image used only for transferable visual style
- automatic hiding of the style image selector when style-reference mode is off
- clean output suitable for direct Krea2 text-to-image generation

## Version notes

See [CHANGELOG.md](CHANGELOG.md) for the current update notes. The latest update clarifies the three target workflows, documents the Krea2 node separately from the Bernini node, and makes optional model-specific imports more tolerant so one missing workflow dependency does not prevent the rest of the node pack from loading.

## Installation

Clone this repository into the ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/liaodaobin/comfyui-wenwu.git
```

Install Python dependencies:

```bash
pip install -r comfyui-wenwu/requirements.txt
```

Restart ComfyUI after installation.

## Dependencies

The prompt generator expects `ComfyUI-llama-cpp` to be installed next to this node directory:

```text
ComfyUI/custom_nodes/ComfyUI-llama-cpp
ComfyUI/custom_nodes/comfyui-wenwu
```

If you want this plugin to be copyable as a single folder, copy both directories together as siblings.

Python dependencies are listed in `requirements.txt`.

The Krea2 node depends on `ComfyUI-llama-cpp` for local prompt rewriting. The ACE Step music node also expects an existing ACE Step 1.5 ComfyUI setup, including the required ACE Step nodes and model files. This repository does not include ACE Step model weights, llama.cpp model weights, or ComfyUI itself.

## Notes

- Runtime logs are written under `logs/` and are ignored by git.
- Python cache files are ignored by git.
- The current repository does not include model weights or ComfyUI itself.
