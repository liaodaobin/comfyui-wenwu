import hashlib
import importlib
import importlib.util
import gc
import json
import os
import re
import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

import comfy.model_management
import folder_paths
from .core_utils import image_tensor_batch_to_data_urls, mie_log
from .bernini_prompts import (
    ADS2V_TEMPLATE,
    I2I_TEMPLATE,
    I2V_TEMPLATE,
    JSON_MODE_TASKS,
    R2I_TEMPLATE,
    R2V_TEMPLATE,
    RI2I_TEMPLATE,
    SYSTEM_PROMPTS,
    TASK_TYPES,
    T2I_A14B_EN_SYS_PROMPT,
    T2V_A14B_EN_SYS_PROMPT,
    VI2V_TEMPLATE,
    VR2V_TEMPLATE,
    V2V_TEMPLATE,
    parse_task_code,
)

MY_CATEGORY = "WenWu/Prompt"
DEFAULT_VIDEO_FRAMES = 3
DEFAULT_VIDEO_MAX_SIZE = 256
DEFAULT_REFERENCE_VIDEO_FRAMES = 24
DEFAULT_BERNINI_LENGTH = 81
MAX_EMBEDDED_REFERENCE_IMAGES = 6
EMBEDDED_REFERENCE_MAX_EDGE = 1024
BERNINI_RESOLUTION_PRESETS = (
    "832*480 横",
    "480*832 竖",
    "1280*720 横",
    "720*1280 竖",
)
BERNINI_RESOLUTION_MAP = {
    "832*480 横": (832, 480),
    "480*832 竖": (480, 832),
    "1280*720 横": (1280, 720),
    "720*1280 竖": (720, 1280),
}


def _extract_json_text(text):
    import ast
    import json

    if not text:
        return text
    s = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    def extract_object(candidate):
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(candidate)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if isinstance(obj, dict) and isinstance(obj.get("rewritten_text"), str):
                return obj["rewritten_text"].strip()
        return None

    extracted = extract_object(s)
    if extracted is not None:
        return extracted
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        extracted = extract_object(m.group(0))
        if extracted is not None:
            return extracted
    return s


def _apply_reference_identity_lock(prompt, image_count):
    """Deterministically prepend a concise identity/outfit lock for reference subjects."""
    count = max(0, min(int(image_count or 0), MAX_EMBEDDED_REFERENCE_IMAGES))
    if count == 0:
        return (prompt or "").strip()
    refs = ", ".join(f"image{i}" for i in range(count))
    lock = (
        f"Use the references ({refs}) only for their intended roles. "
        "Preserve the exact facial identity, apparent age, hairstyle, hair color, skin tone, and body "
        "proportions of every person intended to remain. When another reference supplies clothing, an "
        "object, or a style, transfer only that requested attribute with its exact colors, materials, "
        "and details; do not copy that reference's unrelated person, pose, framing, or background. "
        "Do not swap, blend, redesign, simplify, or substitute identities, and do not add logos, emblems, "
        "or costume details absent from the corresponding reference."
    )
    body = (prompt or "").strip()
    return f"{lock} {body}" if body else lock


def _build_messages(system_prompt, user_text, image_urls, image_detail="auto"):
    parts = []
    for i, url in enumerate(image_urls or []):
        parts.append({"type": "text", "text": f"\n[Image {i}]:"})
        parts.append({"type": "image_url", "image_url": {"url": url, "detail": image_detail}})
    if user_text:
        parts.append({"type": "text", "text": user_text})
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": parts}]


def _sample_urls(urls, n):
    if not urls or n <= 0:
        return []
    if len(urls) <= n:
        return list(urls)
    if n == 1:
        return [urls[len(urls) // 2]]
    idx = [round(i * (len(urls) - 1) / (n - 1)) for i in range(n)]
    out = []
    seen = set()
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(urls[i])
    return out


def _sample_frame_indices(count, max_frames, mode="evenly", stride=1):
    if count <= 0:
        return []
    max_frames = max(1, int(max_frames or DEFAULT_VIDEO_FRAMES))
    stride = max(1, int(stride or 1))
    if mode == "every_nth":
        indices = list(range(0, count, stride))
        if len(indices) > max_frames:
            indices = indices[:max_frames]
        return indices
    if count <= max_frames:
        return list(range(count))
    if max_frames == 1:
        return [count // 2]
    indices = [round(i * (count - 1) / (max_frames - 1)) for i in range(max_frames)]
    out = []
    seen = set()
    for i in indices:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _scale_frame_np(img_np, max_size):
    max_size = int(max_size or 0)
    if max_size <= 0 or img_np.ndim < 3:
        return img_np
    h, w = img_np.shape[:2]
    longest = max(h, w)
    if longest <= max_size:
        return img_np
    scale = max_size / longest
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _resolve_bernini_resolution(resolution):
    return BERNINI_RESOLUTION_MAP.get(str(resolution or "").strip(), BERNINI_RESOLUTION_MAP[BERNINI_RESOLUTION_PRESETS[0]])


def _clean_positive_int(value, default, minimum=1, maximum=8192):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _purge_vram_for_sampling(connector):
    """Release llama/mmproj before CLIP, VAE and Bernini are loaded."""
    try:
        llama_nodes = connector._get_llama_nodes()
        llama_nodes.LLAMA_CPP_STORAGE.clean(all=True)
    except Exception as error:
        mie_log(f"WenWu: llama cleanup failed: {type(error).__name__}: {error}")

    gc.collect()
    comfy.model_management.soft_empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except (RuntimeError, AttributeError):
            pass
    mie_log("WenWu: llama and vision projector unloaded before external CLIP encoding")


def _image_batch_to_sampled_data_urls(image, max_frames, mode="evenly", stride=1, max_size=DEFAULT_VIDEO_MAX_SIZE, fmt=".jpg"):
    if image is None or not hasattr(image, "ndim"):
        return []
    if image.ndim == 3:
        frames = [image]
    elif image.ndim == 4:
        indices = _sample_frame_indices(int(image.shape[0]), max_frames, mode=mode, stride=stride)
        frames = [image[i] for i in indices]
    else:
        return []

    urls = []
    for frame in frames:
        if hasattr(frame, "detach"):
            img_np = frame.detach().cpu().numpy()
        else:
            img_np = np.array(frame)
        img_np = (np.clip(img_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        if len(frames) > 1:
            img_np = _scale_frame_np(img_np, max_size)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(fmt, img_bgr)
        if ok:
            import base64

            urls.append("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8"))
    return urls


def _normalize_runtime_options(
    image_detail,
    temperature,
    top_p,
    max_tokens,
    timeout,
    model_name,
    clear_context,
    video_sample_mode,
    video_frame_stride,
    video_max_size,
):
    image_detail = str(image_detail or "auto")
    if image_detail not in ("auto", "low", "high"):
        image_detail = "auto"

    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        temperature = 0.7
    if not 0.0 <= temperature <= 2.0:
        temperature = 0.7

    try:
        top_p = float(top_p)
    except (TypeError, ValueError):
        top_p = 0.9
    if not 0.0 <= top_p <= 1.0:
        top_p = 0.9

    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        max_tokens = 8192
    if max_tokens < 64:
        max_tokens = 8192

    if timeout not in (30, 60, 120, 300):
        timeout = 30
    if not isinstance(model_name, str) or not model_name:
        model_name = "ComfyUI-llama-cpp"
    clear_context = bool(clear_context)

    video_sample_mode = str(video_sample_mode or "evenly")
    if video_sample_mode not in ("evenly", "every_nth"):
        video_sample_mode = "evenly"
    try:
        video_frame_stride = int(video_frame_stride)
    except (TypeError, ValueError):
        video_frame_stride = 1
    video_frame_stride = max(1, video_frame_stride)
    try:
        video_max_size = int(video_max_size)
    except (TypeError, ValueError):
        video_max_size = DEFAULT_VIDEO_MAX_SIZE
    if video_max_size < 128:
        video_max_size = DEFAULT_VIDEO_MAX_SIZE

    return (
        image_detail,
        temperature,
        top_p,
        max_tokens,
        timeout,
        model_name,
        clear_context,
        video_sample_mode,
        video_frame_stride,
        video_max_size,
    )


class LocalLlamaCppConnector:
    api_url = "comfyui://ComfyUI-llama-cpp"
    api_token = ""
    _THINK_BLOCK_RE = re.compile(
        r"<think>[\s\S]*?</think>|<thinking>[\s\S]*?</thinking>",
        re.IGNORECASE,
    )

    def __init__(self, llama_model, model_name="ComfyUI-llama-cpp", clear_context=True, timeout=30):
        self.llama_model = llama_model
        self.model = model_name or "ComfyUI-llama-cpp"
        self.clear_context = clear_context
        self.timeout = timeout
        self.max_retries = 1
        self.retry_delay = 0

    def get_state(self):
        return f"{self.api_url}{self.model}{self.clear_context}{self.llama_model}"

    def _sanitize_response(self, text, preserve_thinking=False):
        if text is None or preserve_thinking:
            return text
        return self._THINK_BLOCK_RE.sub("", text).strip()

    def _get_llama_nodes(self):
        target = os.path.normcase(os.path.join("custom_nodes", "ComfyUI-llama-cpp", "nodes.py"))
        for module in list(sys.modules.values()):
            module_file = getattr(module, "__file__", None)
            if module_file and os.path.normcase(module_file).endswith(target):
                return module
        for name in ("custom_nodes.ComfyUI-llama-cpp.nodes", "ComfyUI-llama-cpp.nodes"):
            try:
                return importlib.import_module(name)
            except Exception:
                pass
        plugin_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ComfyUI-llama-cpp"))
        if not os.path.exists(os.path.join(plugin_dir, "nodes.py")):
            raise RuntimeError("ComfyUI-llama-cpp is not installed next to comfyui-wenwu.")
        package_name = "_wenwu_comfy_llama_cpp"
        if package_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                package_name,
                os.path.join(plugin_dir, "__init__.py"),
                submodule_search_locations=[plugin_dir],
            )
            package = importlib.util.module_from_spec(spec)
            sys.modules[package_name] = package
            spec.loader.exec_module(package)
        return sys.modules[f"{package_name}.nodes"]

    def invoke(self, messages, **kwargs):
        llama_nodes = self._get_llama_nodes()
        storage = llama_nodes.LLAMA_CPP_STORAGE
        if not storage.llm or storage.current_config != self.llama_model:
            storage.load_model(self.llama_model)

        params = {}
        for key in ("max_tokens", "temperature", "top_p", "top_k", "frequency_penalty", "presence_penalty", "repeat_penalty", "seed"):
            value = kwargs.get(key, None)
            if value is not None:
                params[key] = value

        output = storage.llm.create_chat_completion(messages=messages, stream=False, **params)
        try:
            content = output["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Unexpected llama.cpp response format: {type(e).__name__}. Response: {output}")

        if self.clear_context and storage.llm is not None:
            try:
                storage.llm.n_tokens = 0
                storage.llm._ctx.memory_clear(True)
            except Exception:
                pass
        return self._sanitize_response(content).strip()


def _load_embedded_reference_images(value):
    """Load the filenames serialized by the WenWu thumbnail grid as one IMAGE batch."""
    try:
        filenames = json.loads(value or "[]")
    except (TypeError, ValueError, json.JSONDecodeError):
        filenames = []
    if not isinstance(filenames, list):
        filenames = []

    images = []
    for filename in filenames[:MAX_EMBEDDED_REFERENCE_IMAGES]:
        if not isinstance(filename, str) or not filename.strip():
            continue
        path = folder_paths.get_annotated_filepath(filename)
        if not os.path.isfile(path):
            continue
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            width, height = image.size
            scale = min(1.0, EMBEDDED_REFERENCE_MAX_EDGE / max(width, height))
            if scale < 1.0:
                image = image.resize(
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    Image.Resampling.LANCZOS,
                )
            tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)
            images.append(tensor)

    if not images:
        return torch.empty((0, 64, 64, 3), dtype=torch.float32)

    target_h, target_w = images[0].shape[1:3]
    normalized = []
    for image in images:
        if image.shape[1:3] != (target_h, target_w):
            image = torch.nn.functional.interpolate(
                image.movedim(-1, 1),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).movedim(1, -1)
        normalized.append(image)
    return torch.cat(normalized, dim=0)


def _build_integrated_bernini_conditioning(
    prompt,
    clip,
    vae,
    width,
    height,
    length,
    batch_size,
    source_video=None,
    reference_video=None,
    reference_images=None,
    negative_prompt="bad video, low quality, blurry, distorted",
    use_default_negative=True,
    ref_max_size=848,
):
    if clip is None or vae is None:
        return None, None, None

    from nodes import CLIPTextEncode, NODE_CLASS_MAPPINGS

    negative_text = (negative_prompt or "").strip()
    if not negative_text and use_default_negative:
        negative_text = "bad video, low quality, blurry, distorted, artifacts, deformed"
    encoder = CLIPTextEncode()
    positive = encoder.encode(clip, prompt)[0]
    negative = encoder.encode(clip, negative_text)[0]

    conditioning_class = NODE_CLASS_MAPPINGS.get("BerniniConditioning")
    if conditioning_class is None:
        raise RuntimeError(
            "BerniniConditioning is unavailable. Restart ComfyUI and verify ComfyUI-Bernini loads correctly."
        )
    instance = conditioning_class()
    if hasattr(instance, "apply"):
        kwargs = {
            "source_video": source_video,
            "reference_video": reference_video,
            "ref_max_size": ref_max_size,
        }
        if reference_images is not None and reference_images.shape[0] > 0:
            for index in range(min(reference_images.shape[0], MAX_EMBEDDED_REFERENCE_IMAGES)):
                kwargs[f"reference_image_{index}"] = reference_images[index:index + 1]
        return instance.apply(
            positive,
            negative,
            vae,
            width,
            height,
            length,
            batch_size,
            **kwargs,
        )

    if hasattr(conditioning_class, "execute"):
        reference_map = None
        if reference_images is not None and reference_images.shape[0] > 0:
            reference_map = {
                f"reference_image_{index}": reference_images[index:index + 1]
                for index in range(min(reference_images.shape[0], MAX_EMBEDDED_REFERENCE_IMAGES))
            }
        result = conditioning_class.execute(
            positive,
            negative,
            vae,
            width,
            height,
            length,
            batch_size,
            source_video=source_video,
            reference_video=reference_video,
            reference_images=reference_map,
            ref_max_size=ref_max_size,
        )
        return result.result if hasattr(result, "result") else tuple(result)

    function_name = getattr(conditioning_class, "FUNCTION", None)
    if function_name and hasattr(instance, function_name):
        return getattr(instance, function_name)(
            positive=positive,
            negative=negative,
            vae=vae,
            width=width,
            height=height,
            length=length,
            batch_size=batch_size,
            source_video=source_video,
            reference_video=reference_video,
            ref_max_size=ref_max_size,
        )
    raise RuntimeError("Unsupported BerniniConditioning interface: no apply/execute/FUNCTION method.")


class WenWuPromptGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llama_model": ("LLAMACPPMODEL",),
                "task_type": (list(TASK_TYPES), {"default": "t2i - 文生图"}),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "resolution": (list(BERNINI_RESOLUTION_PRESETS), {"default": BERNINI_RESOLUTION_PRESETS[0]}),
            },
            "optional": {
                "source": ("IMAGE",),
                "reference_images": ("IMAGE",),
                "reference_video": ("IMAGE",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "video_frames": ("INT", {"default": DEFAULT_VIDEO_FRAMES, "min": 1, "max": 1024}),
                "length": ("INT", {"default": DEFAULT_BERNINI_LENGTH, "min": 1, "max": 8192, "step": 4}),
                "reference_video_frames": ("INT", {"default": 0, "min": 0, "max": 1024}),
                "image_detail": (["auto", "low", "high"], {"default": "auto"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 4096, "min": 64, "max": 32768}),
                "timeout": ([30, 60, 120, 300], {"default": 30}),
                "model_name": ("STRING", {"default": "ComfyUI-llama-cpp"}),
                "clear_context": ("BOOLEAN", {"default": True}),
                "purge_vram": ("BOOLEAN", {"default": True}),
                "uploaded_images_json": ("STRING", {"default": "[]"}),
                "custom_width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16, "label": "宽度 width"}),
                "custom_height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16, "label": "高度 height"}),
                "negative_prompt": ("STRING", {"default": "bad video, low quality, blurry, distorted", "multiline": True}),
                "use_default_negative": ("BOOLEAN", {"default": True}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "ref_max_size": ("INT", {"default": 848, "min": 16, "max": 8192, "step": 16}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT", "IMAGE", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("wenwu_prompt", "wenwu_prompt_ref", "width", "height", "length", "reference_images", "positive", "negative", "latent")
    FUNCTION = "generate"
    CATEGORY = MY_CATEGORY

    def _chat(self, connector, system_prompt, user_text, image_urls, json_mode=False, image_detail="auto", temperature=0.7, top_p=0.9, max_tokens=8192):
        messages = _build_messages(system_prompt, user_text, image_urls, image_detail=image_detail)
        mie_log(
            f"WenWu _chat: model={getattr(connector, 'model', '?')} json_mode={json_mode} "
            f"image_detail={image_detail} images={len(image_urls)} system_chars={len(system_prompt or '')} "
            f"user_chars={len(user_text or '')}"
        )
        out = connector.invoke(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        return _extract_json_text(out) if json_mode else out

    def _bilingual_ref(self, connector, chinese_source, english_prompt):
        chinese_source = (chinese_source or "").strip()
        english_prompt = (english_prompt or "").strip()
        chinese_translation = self._translate_to_chinese(connector, english_prompt)
        if english_prompt and chinese_translation:
            return f"{english_prompt}\n{chinese_translation}"
        if english_prompt:
            return english_prompt
        return chinese_source

    def _translate_to_chinese(self, connector, text):
        text = (text or "").strip()
        if not text:
            return text
        translate_prompt = (
            "You are a precise English-to-Chinese translator for image/video prompt editing. "
            "Translate the following English prompt into natural Chinese. "
            "Keep technical terms, scene structure, and prompt semantics intact. "
            "Output only the Chinese translation, no explanation.\n\n"
            f"{text}"
        )
        try:
            zh = self._chat(
                connector,
                "You are a helpful assistant.",
                translate_prompt,
                [],
                json_mode=False,
                image_detail="auto",
                temperature=0.2,
                top_p=1.0,
                max_tokens=2048,
            )
            return (zh or "").strip()
        except Exception:
            return text

    def generate(
        self,
        llama_model,
        task_type,
        user_prompt,
        seed,
        resolution,
        source=None,
        reference_images=None,
        reference_video=None,
        clip=None,
        vae=None,
        uploaded_images_json="[]",
        video_frames=DEFAULT_VIDEO_FRAMES,
        length=DEFAULT_BERNINI_LENGTH,
        reference_video_frames=0,
        image_detail="auto",
        temperature=0.7,
        top_p=0.9,
        max_tokens=4096,
        timeout=30,
        model_name="ComfyUI-llama-cpp",
        clear_context=True,
        purge_vram=True,
        custom_width=832,
        custom_height=480,
        negative_prompt="bad video, low quality, blurry, distorted",
        use_default_negative=True,
        batch_size=1,
        ref_max_size=848,
    ):
        connector = LocalLlamaCppConnector(llama_model, model_name=model_name, clear_context=clear_context, timeout=timeout)
        code = parse_task_code(task_type)
        width, height = _resolve_bernini_resolution(resolution)
        width = _clean_positive_int(custom_width, width, minimum=16, maximum=8192)
        height = _clean_positive_int(custom_height, height, minimum=16, maximum=8192)
        length = _clean_positive_int(length, DEFAULT_BERNINI_LENGTH, minimum=1, maximum=8192)

        embedded_reference_images = _load_embedded_reference_images(uploaded_images_json)
        effective_reference_images = (
            embedded_reference_images
            if embedded_reference_images.shape[0] > 0
            else reference_images
        )

        def finalize(prompt):
            prompt = prompt or user_prompt
            prompt_ref = self._bilingual_ref(connector, user_prompt, prompt)
            if purge_vram:
                _purge_vram_for_sampling(connector)
            output_images = effective_reference_images
            if output_images is None:
                output_images = torch.empty((0, 64, 64, 3), dtype=torch.float32)
            positive, negative, latent = _build_integrated_bernini_conditioning(
                prompt,
                clip,
                vae,
                width,
                height,
                length,
                batch_size,
                source_video=source,
                reference_video=reference_video,
                reference_images=output_images,
                negative_prompt=negative_prompt,
                use_default_negative=use_default_negative,
                ref_max_size=ref_max_size,
            )
            result = (
                prompt,
                prompt_ref,
                width,
                height,
                length,
                output_images,
                positive,
                negative,
                latent,
            )
            return {
                "ui": {"wenwu_prompt": [prompt]},
                "result": result,
            }

        source_urls = _image_batch_to_sampled_data_urls(
            source,
            video_frames,
            max_size=DEFAULT_VIDEO_MAX_SIZE,
        )
        ref_img_urls = _image_batch_to_sampled_data_urls(
            effective_reference_images,
            16,
            mode="evenly",
            max_size=DEFAULT_VIDEO_MAX_SIZE,
        )
        ref_vid_urls = _image_batch_to_sampled_data_urls(
            reference_video,
            reference_video_frames or (DEFAULT_REFERENCE_VIDEO_FRAMES if code == "ads2v" else DEFAULT_VIDEO_FRAMES),
            max_size=DEFAULT_VIDEO_MAX_SIZE,
        )
        if code == "i2v" and not source_urls and len(ref_img_urls) > 1:
            # The built-in gallery contains references, not first/middle/last frames.
            # Multiple gallery images therefore mean reference composition (r2v).
            code = "r2v"
        base_sys = SYSTEM_PROMPTS.get(code, SYSTEM_PROMPTS["default"])
        json_mode = code in JSON_MODE_TASKS

        if code == "t2v":
            out = self._chat(connector, T2V_A14B_EN_SYS_PROMPT, user_prompt, [], json_mode=False, image_detail=image_detail, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            return finalize(out)
        if code == "t2i":
            out = self._chat(connector, T2I_A14B_EN_SYS_PROMPT, user_prompt, [], json_mode=False, image_detail=image_detail, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            return finalize(out)
        if code == "r2v":
            text = R2V_TEMPLATE.format(image_num=max(len(ref_img_urls), 1), original_text=user_prompt)
            out = self._chat(connector, base_sys, text, ref_img_urls, json_mode=True, image_detail=image_detail, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            return finalize(_apply_reference_identity_lock(out, len(ref_img_urls)))
        if code == "r2i":
            text = R2I_TEMPLATE.format(image_num=max(len(ref_img_urls), 1), original_text=user_prompt)
            out = self._chat(connector, base_sys, text, ref_img_urls, json_mode=True, image_detail=image_detail, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            return finalize(_apply_reference_identity_lock(out, len(ref_img_urls)))
        if code == "i2i":
            if not source_urls:
                return finalize(user_prompt)
            text = I2I_TEMPLATE.format(user_prompt=user_prompt)
            out = self._chat(connector, base_sys, text, source_urls[:1] + ref_img_urls, json_mode=False, image_detail=image_detail, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            return finalize(out)
        if code == "i2v":
            if source_urls:
                imgs = source_urls[:1] + ref_img_urls
            elif ref_img_urls:
                imgs = ref_img_urls[:1]
            else:
                return finalize(user_prompt)
            text = I2V_TEMPLATE.format(user_prompt=user_prompt, image_num=len(imgs))
            out = self._chat(connector, base_sys, text, imgs, json_mode=False, image_detail=image_detail, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            return finalize(out)
        if code == "ri2i":
            if source_urls:
                imgs = source_urls + ref_img_urls
                ref_num = len(ref_img_urls)
            elif ref_img_urls:
                imgs = ref_img_urls
                ref_num = max(len(ref_img_urls) - 1, 0)
            else:
                return finalize(user_prompt)
            text = RI2I_TEMPLATE.format(ref_num=ref_num, original_text=user_prompt)
            out = self._chat(connector, base_sys, text, imgs, json_mode=True, image_detail=image_detail, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            return finalize(out)
        if code in ("v2v", "mv2v"):
            text = V2V_TEMPLATE.format(user_prompt=user_prompt)
            out = self._chat(connector, base_sys, text, source_urls, json_mode=False, image_detail=image_detail, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            return finalize(out)
        if code == "ads2v":
            text = ADS2V_TEMPLATE.format(user_prompt=user_prompt)
            out = self._chat(connector, base_sys, text, source_urls + ref_vid_urls, json_mode=False, image_detail=image_detail, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            return finalize(out)
        if code == "vi2v":
            text = VI2V_TEMPLATE.format(user_prompt=user_prompt, image_num=len(ref_img_urls))
            out = self._chat(connector, base_sys, text, source_urls + ref_img_urls, json_mode=False, image_detail=image_detail, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            return finalize(out)
        if code in ("rv2v", "vrc2v"):
            ref_total = ref_img_urls + ref_vid_urls
            text = VR2V_TEMPLATE.format(image_num=max(len(ref_total), 1), original_text=user_prompt)
            out = self._chat(connector, base_sys, text, source_urls + ref_total, json_mode=True, image_detail=image_detail, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            return finalize(out)

        return finalize(user_prompt)

    def is_changed(
        self,
        llama_model,
        task_type,
        user_prompt,
        seed,
        resolution,
        source=None,
        reference_images=None,
        reference_video=None,
        clip=None,
        vae=None,
        uploaded_images_json="[]",
        video_frames=DEFAULT_VIDEO_FRAMES,
        length=DEFAULT_BERNINI_LENGTH,
        reference_video_frames=0,
        image_detail="auto",
        temperature=0.7,
        top_p=0.9,
        max_tokens=4096,
        timeout=30,
        model_name="ComfyUI-llama-cpp",
        clear_context=True,
        purge_vram=True,
        custom_width=832,
        custom_height=480,
        negative_prompt="bad video, low quality, blurry, distorted",
        use_default_negative=True,
        batch_size=1,
        ref_max_size=848,
    ):
        h = hashlib.md5()
        for value in (
            task_type,
            resolution,
            user_prompt,
            seed,
            video_frames,
            length,
            reference_video_frames,
            image_detail,
            temperature,
            top_p,
            max_tokens,
            timeout,
            model_name,
            clear_context,
            purge_vram,
            uploaded_images_json,
            custom_width,
            custom_height,
            negative_prompt,
            use_default_negative,
            batch_size,
            ref_max_size,
        ):
            h.update(str(value).encode("utf-8"))
        try:
            h.update(llama_model.get_state().encode("utf-8"))
        except AttributeError:
            h.update(str(llama_model).encode("utf-8"))
        for src in (source, reference_images, reference_video):
            urls = image_tensor_batch_to_data_urls(src)
            h.update(str(len(urls)).encode("utf-8"))
            if urls:
                h.update(urls[0][:64].encode("utf-8"))
        return h.hexdigest()


class WenWuShowAndSaveAnything:
    LOG_DIR_NAME = "logs"
    LOG_DEFAULT_NAME = "wenwu_show.log"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any_value": ("*"),
                "save_to_log": ("BOOLEAN", {"default": True}),
                },
            "optional": {
                "log_file_name": ("STRING", {"default": cls.LOG_DEFAULT_NAME}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = MY_CATEGORY

    def execute(self, any_value, save_to_log=True, log_file_name="wenwu_show.log", unique_id=None, extra_pnginfo=None):
        text = str(any_value)
        if save_to_log:
            try:
                log_dir = os.path.join(os.path.dirname(__file__), self.LOG_DIR_NAME)
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, log_file_name or "wenwu_show.log")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(text)
                    f.write("\n")
            except Exception:
                pass
        return {"ui": {"text": [text]}, "result": (text,)}
