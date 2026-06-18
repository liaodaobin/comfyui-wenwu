from .wenwu_prompt_generator import WenWuPromptGenerator


class WenWuSimpleTextNode:
    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "WenWu/Prompt"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def noop(self):
        return {}


NODE_CLASS_MAPPINGS = {
    "WenWuPromptGenerator": WenWuPromptGenerator,
    "WenWuSimpleTextNode": WenWuSimpleTextNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WenWuPromptGenerator": "Bernini Prompt wenwu",
    "WenWuSimpleTextNode": "Simple Text",
}

WEB_DIRECTORY = "./js"
