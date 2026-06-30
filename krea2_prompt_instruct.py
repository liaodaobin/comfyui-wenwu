import importlib.util
import os
import sys
from pathlib import Path

import folder_paths
from nodes import LoadImage


LLAMA_CPP_DIR = Path(__file__).resolve().parents[1] / "ComfyUI-llama-cpp"
LLAMA_CPP_INIT = LLAMA_CPP_DIR / "__init__.py"
LLAMA_CPP_NODES = LLAMA_CPP_DIR / "nodes.py"
LLAMA_CPP_ALIAS = "wenwu_krea2_llama_cpp"


def _load_llama_cpp_instruct():
    if not LLAMA_CPP_NODES.exists():
        raise RuntimeError(
            "ComfyUI-llama-cpp was not found next to comfyui-wenwu. "
            "Install it under ComfyUI/custom_nodes first."
        )

    module_name = f"{LLAMA_CPP_ALIAS}.nodes"
    if module_name in sys.modules:
        return sys.modules[module_name].llama_cpp_instruct_adv

    package_spec = importlib.util.spec_from_file_location(
        LLAMA_CPP_ALIAS,
        LLAMA_CPP_INIT,
        submodule_search_locations=[str(LLAMA_CPP_DIR)],
    )
    package = importlib.util.module_from_spec(package_spec)
    sys.modules[LLAMA_CPP_ALIAS] = package

    spec = importlib.util.spec_from_file_location(module_name, LLAMA_CPP_NODES)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.llama_cpp_instruct_adv


KREA2_TEXT_SYSTEM = """You are a Krea2 prompt specialist. Convert the user's Chinese or mixed-language image request into one single continuous English prompt for Krea2 text-to-image generation. Output only the final English prompt, with no title, no explanation, no markdown, no bullet points, no parameters, no Chinese, and no extra commentary. The prompt should be detailed, commercially usable, and optimized for Krea2: lock the main subject first, then enrich scene, composition, camera perspective, lighting, material texture, color system, mood, rendering quality, and visual restrictions. Keep the user's core subject and intent unchanged. If the user request is short, intelligently expand it into a rich professional prompt. Avoid text, logos, watermarks, distorted anatomy, extra limbs, blurry details, clutter, overexposure, underexposure, low resolution, pixelation, ugly deformation, and random floating objects."""


KREA2_STYLE_SYSTEM = """You are a Krea2 single-image style transfer prompt specialist. The input image is only a visual style reference, not content to copy. Convert the user's new subject and scene request into one single continuous English prompt for Krea2. Output only the final English prompt, with no title, no explanation, no markdown, no bullet points, no parameters, no Chinese, and no extra commentary. Preserve the user's new subject, new scene, and core creative logic. Extract only transferable visual qualities from the reference image: color palette, saturation, contrast, lighting softness, shadow logic, material texture, lens mood, depth, atmosphere, grain, and emotional tone. Do not copy the reference image's original objects, people, clothing, symbols, text, layout, or story content. If the user's color or style words conflict with the reference image, silently harmonize them toward the reference style. Include concrete visual descriptors, high-definition rendering quality, clean composition, and restrictions against text, logos, watermarks, distorted anatomy, extra limbs, blurry details, clutter, overexposure, underexposure, pixelation, low-resolution artifacts, and readable symbols."""


DEFAULT_LLAMA_PARAMETERS = {
    "max_tokens": 1024,
    "top_k": 30,
    "top_p": 0.9,
    "min_p": 0.05,
    "typical_p": 1.0,
    "temperature": 0.8,
    "repeat_penalty": 1.0,
    "frequency_penalty": 0.0,
    "present_penalty": 0.0,
    "mirostat_mode": 0,
    "mirostat_eta": 0.1,
    "mirostat_tau": 5.0,
    "state_uid": -1,
}


def _build_text_prompt(user_prompt):
    return "#Krea2 high-end prompt generation\nUser request:\n" + (user_prompt or "").strip()


def _build_style_prompt(user_prompt):
    return (
        "#Krea2 single-image precise style transfer\n"
        "Use the attached image only as a style reference. Generate a Krea2 prompt for this new request:\n"
        + (user_prompt or "").strip()
    )


def _clean_int(value, default, minimum=None, maximum=None):
    try:
        value = int(value)
    except (TypeError, ValueError, OverflowError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _clean_bool(value, default=False):
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    if value is None:
        return default
    return bool(value)


class WenWuKrea2PromptInstruct:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        files = sorted(files) or [""]

        return {
            "required": {
                "llama_model": ("LLAMACPPMODEL",),
                "user_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Enter the image request. Default mode is Krea2 text-to-image prompt generation.",
                }),
                "style_reference": ("BOOLEAN", {
                    "default": False,
                    "label_on": "风格参考",
                    "label_off": "文生图",
                }),
                "style_image": (files, {"image_upload": True, "label": "风格参考图"}),
                "max_frames": ("INT", {"default": 24, "min": 2, "max": 1024, "step": 1}),
                "max_size": ("INT", {"default": 256, "min": 128, "max": 16384, "step": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "force_offload": ("BOOLEAN", {"default": True}),
                "save_states": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
            "optional": {
                "queue_handler": ("*",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("output", "output_list", "state_uid")
    OUTPUT_IS_LIST = (False, True, False)
    FUNCTION = "process"
    CATEGORY = "WenWu/Prompt"

    def process(
        self,
        llama_model,
        user_prompt,
        style_reference,
        style_image,
        max_frames,
        max_size,
        seed,
        force_offload,
        save_states,
        unique_id,
        queue_handler=None,
        parameters=None,
        images=None,
        **kwargs,
    ):
        instruct_cls = _load_llama_cpp_instruct()
        instruct = instruct_cls()

        use_style = bool(style_reference)
        system_prompt = KREA2_STYLE_SYSTEM if use_style else KREA2_TEXT_SYSTEM
        custom_prompt = _build_style_prompt(user_prompt) if use_style else _build_text_prompt(user_prompt)
        max_frames = _clean_int(max_frames, 24, 2, 1024)
        max_size = _clean_int(max_size, 256, 128, 16384)
        seed = _clean_int(seed, 0, 0, 0xffffffffffffffff)
        force_offload = _clean_bool(force_offload, True)
        save_states = _clean_bool(save_states, False)
        embedded_image = LoadImage().load_image(style_image)[0] if use_style and style_image else None
        merged_parameters = dict(DEFAULT_LLAMA_PARAMETERS)
        if isinstance(parameters, dict):
            merged_parameters.update(parameters)

        return instruct.process(
            llama_model=llama_model,
            preset_prompt="Normal - Describe",
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            inference_mode="one by one",
            max_frames=max_frames,
            max_size=max_size,
            seed=seed,
            force_offload=force_offload,
            save_states=save_states,
            unique_id=unique_id,
            parameters=merged_parameters,
            images=embedded_image if use_style else None,
            queue_handler=queue_handler,
        )

    @classmethod
    def VALIDATE_INPUTS(cls, style_reference, style_image, **kwargs):
        if not style_reference:
            return True
        if not style_image:
            return "Style reference is enabled, but no style image is selected."
        if not folder_paths.exists_annotated_filepath(style_image):
            return f"Invalid style image file: {style_image}"
        return True
