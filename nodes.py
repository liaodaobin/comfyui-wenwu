from .wenwu_prompt_generator import WenWuPromptGenerator

try:
    from .ltx23_wenwu_node import WenWuLTX23PromptEnhancer
except Exception as exc:
    WenWuLTX23PromptEnhancer = None
    _LTX23_IMPORT_ERROR = exc

try:
    from .krea2_prompt_instruct import WenWuKrea2PromptInstruct
except Exception as exc:
    WenWuKrea2PromptInstruct = None
    _KREA2_IMPORT_ERROR = exc

try:
    from .ace_step_llm_wenwu import (
        NODE_CLASS_MAPPINGS as ACE_STEP_LLM_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as ACE_STEP_LLM_DISPLAY_NAME_MAPPINGS,
    )
except Exception as exc:
    ACE_STEP_LLM_CLASS_MAPPINGS = {}
    ACE_STEP_LLM_DISPLAY_NAME_MAPPINGS = {}
    _ACE_STEP_IMPORT_ERROR = exc


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
if WenWuLTX23PromptEnhancer is not None:
    NODE_CLASS_MAPPINGS["WenWuLTX23PromptEnhancer"] = WenWuLTX23PromptEnhancer
if WenWuKrea2PromptInstruct is not None:
    NODE_CLASS_MAPPINGS["WenWuKrea2PromptInstruct"] = WenWuKrea2PromptInstruct
NODE_CLASS_MAPPINGS.update(ACE_STEP_LLM_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {
    "WenWuPromptGenerator": "Bernini Prompt wenwu",
    "WenWuSimpleTextNode": "Simple Text",
}
if WenWuLTX23PromptEnhancer is not None:
    NODE_DISPLAY_NAME_MAPPINGS["WenWuLTX23PromptEnhancer"] = "WenWu LTX-2.3 Prompt Enhancer"
if WenWuKrea2PromptInstruct is not None:
    NODE_DISPLAY_NAME_MAPPINGS["WenWuKrea2PromptInstruct"] = "WenWu Krea2 Prompt Instruct"
NODE_DISPLAY_NAME_MAPPINGS.update(ACE_STEP_LLM_DISPLAY_NAME_MAPPINGS)

WEB_DIRECTORY = "./js"




