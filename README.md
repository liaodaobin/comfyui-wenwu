# comfyui-wenwu

ComfyUI custom nodes for Bernini prompt workflows.

This node package is built specifically for Bernini prompt generation inside ComfyUI. Bernini has high requirements for prompt quality, so this node lets users enter natural spoken-style descriptions and then uses a local `llama.cpp` inference workflow through `ComfyUI-llama-cpp` to convert them into professional Bernini prompts.

The goal is to save users from manually writing strict Bernini prompt text while keeping the conversion local.

## What this node provides

- **Bernini Prompt wenwu**: converts natural user descriptions into professional Bernini-oriented prompt text for image and video tasks through a local `ComfyUI-llama-cpp` model.
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

## Notes

- Runtime logs are written under `logs/` and are ignored by git.
- Python cache files are ignored by git.
- The current repository does not include model weights or ComfyUI itself.
