from .wenwu_prompt_generator import WenWuPromptGenerator
from .ltx23_wenwu_node import WenWuLTX23PromptEnhancer
from .krea2_prompt_instruct import WenWuKrea2PromptInstruct
from .ace_step_llm_wenwu import (
    NODE_CLASS_MAPPINGS as ACE_STEP_LLM_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as ACE_STEP_LLM_DISPLAY_NAME_MAPPINGS,
)
from .ace_step_remix_wenwu import (
    NODE_CLASS_MAPPINGS as ACE_STEP_REMIX_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as ACE_STEP_REMIX_DISPLAY_NAME_MAPPINGS,
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
    "WenWuLTX23PromptEnhancer": WenWuLTX23PromptEnhancer,
    "WenWuKrea2PromptInstruct": WenWuKrea2PromptInstruct,
    "WenWuSimpleTextNode": WenWuSimpleTextNode,
}
NODE_CLASS_MAPPINGS.update(ACE_STEP_LLM_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(ACE_STEP_REMIX_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {
    "WenWuPromptGenerator": "Bernini Prompt wenwu",
    "WenWuLTX23PromptEnhancer": "WenWu LTX-2.3 Prompt Enhancer",
    "WenWuKrea2PromptInstruct": "WenWu Krea2 Prompt Instruct",
    "WenWuSimpleTextNode": "Simple Text",
}
NODE_DISPLAY_NAME_MAPPINGS.update(ACE_STEP_LLM_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(ACE_STEP_REMIX_DISPLAY_NAME_MAPPINGS)

WEB_DIRECTORY = "./js"




