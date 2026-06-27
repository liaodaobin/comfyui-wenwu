from .wenwu_prompt_generator import WenWuPromptGenerator
from .ace_step_llm_wenwu import (
    NODE_CLASS_MAPPINGS as ACE_STEP_LLM_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as ACE_STEP_LLM_DISPLAY_NAME_MAPPINGS,
)


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
NODE_CLASS_MAPPINGS.update(ACE_STEP_LLM_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {
    "WenWuPromptGenerator": "Bernini Prompt wenwu",
    "WenWuSimpleTextNode": "Simple Text",
}
NODE_DISPLAY_NAME_MAPPINGS.update(ACE_STEP_LLM_DISPLAY_NAME_MAPPINGS)

WEB_DIRECTORY = "./js"
