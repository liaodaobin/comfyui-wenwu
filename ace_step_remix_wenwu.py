import importlib.util
import os
import sys


_WHISPER_NODE = None


def _get_whisper_node():
    """Reuse ComfyUI-Whisper so its model cache and VRAM offload still work."""
    global _WHISPER_NODE
    if _WHISPER_NODE is not None:
        return _WHISPER_NODE

    for module in list(sys.modules.values()):
        node = getattr(module, "ApplyWhisperNode", None)
        if node is not None:
            _WHISPER_NODE = node
            return node

    here = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(os.path.dirname(here), "ComfyUI-Whisper", "apply_whisper.py")
    if not os.path.isfile(module_path):
        raise RuntimeError("未找到 ComfyUI-Whisper。请先安装并启用该插件。")
    spec = importlib.util.spec_from_file_location("wenwu_comfyui_whisper", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _WHISPER_NODE = module.ApplyWhisperNode
    return _WHISPER_NODE


def _display_language(code):
    names = {
        "auto": "auto", "zh": "Chinese", "en": "English", "ja": "Japanese",
        "ko": "Korean", "yue": "Cantonese", "fr": "French", "de": "German",
        "es": "Spanish", "ru": "Russian",
    }
    return names.get(code, "auto")


class AceStepSourceLyrics:
    """Transcribe a source song and expose line-broken lyrics for ACE-Step remixing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "whisper_model": (["large-v3", "large-v3-turbo", "turbo", "medium", "small"], {"default": "large-v3"}),
                "language": (["auto", "zh", "en", "ja", "ko", "yue", "fr", "de", "es", "ru"], {"default": "auto"}),
            },
            "optional": {
                "hint": ("STRING", {"default": "这是歌曲歌词，请保留重复段落并尽量按演唱停顿断行。", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("lyrics", "raw_transcript")
    FUNCTION = "transcribe"
    CATEGORY = "AceStep/Remix"
    OUTPUT_NODE = True

    def transcribe(self, audio, whisper_model, language, hint=""):
        whisper_node = _get_whisper_node()()
        raw_text, segments, _ = whisper_node.apply_whisper(
            audio, whisper_model, _display_language(language), hint or ""
        )
        lines = [str(item.get("value", "")).strip() for item in (segments or [])]
        lyrics = "\n".join(line for line in lines if line) or str(raw_text or "").strip()
        return {
            "ui": {"text": ["[识别歌词]\n" + lyrics]},
            "result": (lyrics, str(raw_text or "").strip()),
        }


NODE_CLASS_MAPPINGS = {
    "AceStepSourceLyrics": AceStepSourceLyrics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepSourceLyrics": "ACE Step 源歌曲歌词识别（文武）",
}
