from .wenwu_prompt_generator import WenWuPromptGenerator, WenWuShowAndSaveAnything

NODE_CLASS_MAPPINGS = {
    "WenWuPromptGenerator": WenWuPromptGenerator,
    "WenWuShowAndSaveAnything": WenWuShowAndSaveAnything,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WenWuPromptGenerator": "Bernini Prompt wenwu",
    "WenWuShowAndSaveAnything": "WenWu Show And Save Anything",
}

WEB_DIRECTORY = "./js"
