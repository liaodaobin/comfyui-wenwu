# comfyui-wenwu

ComfyUI custom nodes for WenWu prompt workflows.

This node package is built for local prompt generation inside ComfyUI. It lets users enter natural spoken-style descriptions and then uses local `llama.cpp` inference through `ComfyUI-llama-cpp` to convert them into structured prompts.

The goal is to save users from manually writing strict prompt text while keeping the conversion local.

## What this node provides

- **Bernini Prompt wenwu**: converts natural user descriptions into professional Bernini-oriented prompt text for image and video tasks through a local `ComfyUI-llama-cpp` model.
- **ACE Step 1.5 LLM WENWU**: converts natural music descriptions into ACE Step 1.5-ready tags, lyrics, BPM, duration, language, key, and conditioning. It includes a preview/refresh UI and an editable confirmed prompt box.
- **Simple Text**: a lightweight canvas text node that can be edited by double-clicking the node or using the node context menu.

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

Python dependencies are listed in `requirements.txt`.

The ACE Step music node also expects an existing ACE Step 1.5 ComfyUI setup, including the required ACE Step nodes and model files. This repository does not include ACE Step model weights, llama.cpp model weights, or ComfyUI itself.

## Notes

- Runtime logs are written under `logs/` and are ignored by git.
- Python cache files are ignored by git.
- The current repository does not include model weights or ComfyUI itself.
