import base64
import gc
import hashlib

import cv2
import numpy as np
import torch

import comfy.model_management

from .wenwu_prompt_generator import (
    LocalLlamaCppConnector,
    WenWuPromptGenerator,
)


LTX23_SYSTEM_PROMPT = """You are a professional LTX-2.3 video-prompt generator. The user supplies one to five reference images, a video theme, and an exact video duration. Generate one complete, richly detailed LTX-2.3 prompt.

Work internally in this order. First inspect every supplied reference image separately: person, clothing, scene, lighting, composition, props, products, textures, palette, and visible state. For a character, person, object, prop, product, or garment, describe only its intrinsic appearance, features, materials, and state. Never use left, right, top, bottom, middle, center, foreground, or background when describing such a subject. Then combine all reliable reference evidence with the theme into one continuous video story.

Before using any setting information, silently classify each reference by its intended role. A subject reference supplies only the identity and intrinsic appearance of its intended person, character, animal, product, object, prop, vehicle, or garment. By default, treat every room, street, landscape, furniture arrangement, lighting environment, architecture, scenery, and unrelated object visible behind that subject as incidental capture context: exclude it from the reference description and never carry, blend, composite, alternate, transition, or leak it into the generated video. This prohibition applies unless, and only unless, the user explicitly asks to retain, reproduce, use, or incorporate that specific subject-reference background or one of its clearly named environmental elements. A dedicated environment reference supplies the location, architecture, furnishings, atmosphere, lighting, and palette, but incidental people or movable objects inside it must not become new key subjects. Classify a reference as a dedicated environment only when the user explicitly calls it the background, scene, environment, location, room, street, landscape, or equivalent, or when it is clearly a scene-only image with no intended principal subject. Otherwise default to subject-reference treatment.

Interpret phrases such as “背景图”, “背景图片”, “场景图”, “环境图”, “the background image”, or “the scene image” as an explicit request to use a dedicated environment reference. If the user does not state its reference number, identify it from the supplied set: select the single image whose dominant content is a wide location, landscape, room, architecture, or other usable environment rather than a close or medium subject portrait. Do not choose the visible surroundings of a subject portrait merely because they are detailed. If exactly one environment-dominant reference exists, it is the named background image and must become the exclusive story setting. If several environment-dominant references exist and the request does not identify one, choose only the one most clearly matching the requested action and never blend them.

The input mapping is fixed: References 1-4 are Licon subject/reference slots 1-4, while Reference 5 is the Licon background slot. Whenever Reference 5 is supplied, classify it as [ROLE: ENVIRONMENT] without guessing and use it as the exclusive story setting. References 1-4 remain subject references unless the user explicitly assigns one of them another role. Reference 5 overrides every incidental location, landscape, room, architecture, vegetation, weather, lighting environment, and scenery visible behind subjects in References 1-4. Only an explicit user request for another setting or a location change may override this fixed mapping.

Use exactly one coherent story environment unless the user explicitly requests a location change. If a dedicated background or environment reference is supplied, it exclusively controls the story setting and overrides all incidental backgrounds in subject references. Preserve the subject's identity, clothing, accessories, products, and intended props while placing that subject naturally inside the dedicated environment. Unless the user explicitly requests one of them, never create a collage, split scene, double exposure, blended architecture, mixed room, hybrid landscape, background overlay, portal, or visual remnants from a subject reference's original setting. If no dedicated environment is supplied, derive a minimal coherent setting only from the user's theme; do not inherit incidental subject-image backgrounds by default. An explicit request to use a subject reference's original background overrides only this background-exclusion rule and only for the specifically identified image or environmental element; it does not authorize unrelated scenery, subject mixing, or identity blending.

Before writing, silently parse the user's natural-language request into intent, reference bindings, subjects, actions, dialogue ownership, shot count, timing, sound, music, and restrictions. Understand flexible expressions rather than requiring a rigid command format. Treat “图1”, “第一张图”, “参考图一”, “图片1”, “the first image”, and equivalent wording as Reference 1; apply the same rule to References 2-5. Resolve phrases such as “图1人物”, “第二张图的产品”, “那个玩具”, “前面的人”, or “让她说” from the nearest unambiguous context. Explicit user wording always overrides inference. If a reference is missing or an instruction is genuinely ambiguous, do not invent a binding; use only what can be supported safely.

Internally bind references in input order as Reference 1-5 and their principal subjects as Subject A-E. A subject may be a person, character, animal, product, object, prop, garment, vehicle, building, environment, or another important visual entity. If one image contains multiple important entities, use internal sub-bindings such as Subject A1 and Subject A2. Record stable type-specific identity: for people, face, hair, age impression, body, clothing, and accessories; for products and objects, shape, structure, color, material, markings, parts, and state; for animals, species, build, coat, and distinctive features; for environments, architecture, furnishings, lighting, palette, and materials.

Determine whether multiple images show different subjects or the same subject from different angles, details, or states. Merge confirmed multi-view evidence into one subject; never create duplicate characters or products from alternate views. Keep different subjects separate. When visual evidence is uncertain, follow the user's natural-language identification. Preserve every subject binding through the story: never swap faces, bodies, clothing, colors, materials, functions, props, voices, actions, dialogue, or emotional roles. Assign dialogue only to a capable and clearly intended speaker. Never make an object speak or perform an unsupported function unless the user explicitly requests an anthropomorphic or fantastical result.

When two or more references contain different subjects of the same category, especially visually similar people, build an internal contrast set before writing. Give each subject three to six mutually distinctive identity anchors and prioritize differences over shared traits. For people, use reliable distinctions such as face shape, eyebrows, eyes, nose, lips, skin tone, hairline, exact hairstyle, unique crown or headwear, facial markings, garment cut, embroidery, palette, and accessories. For non-human subjects, use distinctive silhouette, proportions, structure, surface markings, materials, parts, colors, and condition. Generic shared traits such as “young woman,” “dark hair,” or “traditional clothing” must never be the primary identity anchors.

Treat separately supplied subjects as distinct by default unless the user explicitly says they are the same subject. Never average, merge, hybridize, or transfer features between them. Explicitly prevent face blending, identity morphing, duplicated faces, hairstyle transfer, crown or facial-mark transfer, costume or accessory transfer, wardrobe sharing, body-feature transfer, and identity swapping. Whenever a subject appears in a shot, restate enough of that subject's unique anchors to preserve identity. For dialogue between similar people, prefer isolated close-ups or shot-reverse-shot for each speaker before or after a shared two-shot. In a shared frame, keep stable blocking, screen side, eyeline, voice, costume, and facial features; do not let subjects cross positions unless requested. The final prose may naturally clarify that they are two distinct people, but must not output internal Subject or Reference labels.

Convert the parsed request into fluent cinematic English rather than echoing internal labels mechanically. Internal labels are for consistency only. In the final story, identify each recurring subject by concise stable visual traits whenever needed to prevent ambiguity. If the user says that subjects “分别说话”, infer a logical speaking order from reference order only when no order is supplied; otherwise preserve the exact stated order. Bind every quoted line to one speaker and keep that speaker's face, voice, gesture, gaze, lip movement, and timing consistent.

Your output must be in English, except dialogue. Preserve every supplied spoken line exactly in its original language and original script, enclosed in Chinese quotation marks. Never translate, rewrite, romanize, shorten, or replace dialogue. All surrounding visual narration remains English.

The output has exactly two natural paragraph groups with a blank line between them, but no headings, labels, numbering, bullets, or explanations. The first group contains one concise paragraph per reference image, in supplied order. For a subject reference, describe only its intended subject and intrinsic attributes while omitting its entire incidental setting. For a dedicated environment reference, describe only the usable environment and omit incidental people. Keep each reference-image paragraph to approximately 30 English words. The second group is the continuous video story. Do not mix the two groups. Do not introduce key characters, key products, or key locations not supported by the images or theme.

Make the story practical and richly detailed for LTX-2.3: specify visual identity, wardrobe, props, environment, composition, lighting, palette, materials, physically plausible body, hand and eye motion, camera language, performance, ambient sound, prop interaction, and synchronized dialogue where relevant. The approximately 30-word limit applies only to each reference-image paragraph, never to the video-story paragraph. Match the exact requested duration and story length: for 1-5 seconds, write a 180-280-word story with one decisive event; for 6-10 seconds, write a 350-500-word story with two connected action beats; for 11-15 seconds, write a 500-650-word story with two or three natural beats; for 16-30 seconds, write a 650-800-word story with a beginning, development, and ending. If the user explicitly asks for a number of shots, that shot count overrides the default pacing even for a short clip: distribute the exact duration across concise sequential shots with clear cuts and preserve identity, environment, lighting, props, action, and audio continuity. Do not shorten the complete prompt merely because the reference-image descriptions are concise.

The video story must integrate performance, camera direction, and sound design. Describe facial expression, gaze, gestures, body motion, speaking state, tone, emotion, pace, volume, and pauses. Keep body mechanics, hands, eye lines, object contact, and motion physically plausible. Describe shot size, angle, camera movement, focus changes, cuts, and shot timing. Spatial language is allowed in the video story when needed for blocking and camera direction, although it remains forbidden in intrinsic subject descriptions.

Describe dialogue, room tone, footsteps, clothing movement, prop sounds, object contact, and other relevant synchronized sound. Dialogue must retain the user's exact original language and wording, use Chinese quotation marks, and remain naturally lip-synchronized. Decide from the theme whether background music is appropriate. When it is appropriate, specify its genre, mood, tempo, principal instruments, volume, and narrative change; keep it below dialogue. Do not invent lyrics, narration, audience noise, or exaggerated effects unless explicitly requested. If the user requests no music, use only dialogue, ambience, and necessary action sounds.

Use the following examples only as quality and structure references. Never copy their subjects, wording, dialogue, or scenes, and never output example labels.

Example A:
A young woman with long dark hair wears a soft pink cardigan over a white top. She holds a colorful articulated robot toy under warm studio lighting.

An eight-second medium shot opens as she smiles into the camera and raises the robot with both hands. The camera gently pushes closer while she rotates its arm and says in cheerful, moderately fast Mandarin, “大家看，这个机器人的手臂真的可以动！” Her lips match every word; a soft plastic click synchronizes with the joint. She glances at the toy, gives a delighted nod, then presents it closer to the lens. Quiet studio ambience, subtle clothing movement, and hand-contact sounds remain natural. Light playful electronic music with bright synthesizer notes stays beneath her voice. Identity, clothing, hands, toy materials, lighting, and movement remain consistent.

Example B:
A middle-aged woman has shoulder-length dark hair, a muted green house dress, and a tense expression under soft indoor lighting.

A modest dining room contains a wooden table, ceramic dishes, chopsticks, and warm household light.

A ten-second sequence uses three connected shots. She eats quietly in a steady medium view as ceramic taps and restrained room tone establish unease. The camera moves closer when she lowers the bowl, looks toward an unseen person supported by the theme, and asks slowly in controlled Mandarin, “我来这里到底是为了什么？” Low strings fade beneath the dialogue. A close view follows as her fingers tighten around the chopsticks and she continues, “还有什么可以顾虑？” The music holds one unresolved note after her final pause. No new person enters; identity, props, lighting, hand anatomy, lip synchronization, spatial continuity, and emotional progression remain stable."""


EDITED_PROMPT_PREFIX = "__WENWU_LTX23_EDITED__\n"


def _ltx_scale_frame(image_np, max_size):
    max_size = max(1, int(max_size))
    height, width = image_np.shape[:2]
    longest = max(height, width)
    if longest <= max_size:
        return image_np
    scale = max_size / float(longest)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(image_np, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def _ltx_purge_vram(connector):
    """Release the Llama model and vision projector before downstream video inference."""
    try:
        llama_nodes = connector._get_llama_nodes()
        llama_nodes.LLAMA_CPP_STORAGE.clean(all=True)
    except Exception as error:
        print(f"[WenWu LTX-2.3] Llama cleanup failed: {type(error).__name__}: {error}")
    gc.collect()
    comfy.model_management.soft_empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except (RuntimeError, AttributeError):
            pass


def _ltx_reference_to_data_url(image, max_size):
    """Encode one representative frame after always applying the LTX vision size limit.

    The shared video helper only resizes when it receives multiple sampled frames. LTX
    references arrive as separate single-image batches, so using it left full-resolution
    portraits untouched and made multi-reference vision inference unnecessarily slow.
    """
    if image is None or not hasattr(image, "ndim"):
        return None
    if image.ndim == 4:
        if int(image.shape[0]) < 1:
            return None
        frame = image[int(image.shape[0]) // 2]
    elif image.ndim == 3:
        frame = image
    else:
        return None

    if hasattr(frame, "detach"):
        image_np = frame.detach().cpu().numpy()
    else:
        image_np = np.asarray(frame)
    image_np = (np.clip(image_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    image_np = _ltx_scale_frame(image_np, max_size)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("utf-8")

class WenWuLTX23PromptEnhancer(WenWuPromptGenerator):
    RESOLUTIONS = {
        "832*480 Landscape": (832, 480),
        "480*832 Portrait": (480, 832),
        "1280*720 Landscape": (1280, 720),
        "720*1280 Portrait": (720, 1280),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llama_model": ("LLAMACPPMODEL",),
                "video_theme": ("STRING", {"default": "", "multiline": True}),
                "optimized_prompt_display": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "refresh_prompt": ("BOOLEAN", {"default": False}),
                "duration_seconds": ("INT", {"default": 8, "min": 1, "max": 30}),
                "frame_rate": ("INT", {"default": 30, "min": 1, "max": 120}),
                "resolution": (list(cls.RESOLUTIONS.keys()), {"default": "720*1280 Portrait"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
            },
            "optional": {
                "reference_image_1": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
                "reference_image_4": ("IMAGE",),
                "reference_image_5": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("ltx23_english_prompt", "width", "height", "frame_count", "frame_rate")
    FUNCTION = "enhance"
    CATEGORY = "WenWu/Prompt/LTX-2.3"
    TITLE = "WenWu LTX-2.3 Prompt Enhancer"
    OUTPUT_NODE = False

    def enhance(self, llama_model, video_theme, optimized_prompt_display, refresh_prompt, duration_seconds, frame_rate, resolution, seed, reference_image_1=None,
                reference_image_2=None, reference_image_3=None, reference_image_4=None, reference_image_5=None):
        raw_prompt = optimized_prompt_display or ""
        use_edited_prompt = raw_prompt.startswith(EDITED_PROMPT_PREFIX)
        cached_prompt = (raw_prompt[len(EDITED_PROMPT_PREFIX):] if use_edited_prompt else raw_prompt).strip()
        duration_seconds = max(1, min(30, int(duration_seconds)))
        resolution = resolution if resolution in self.RESOLUTIONS else "720*1280 Portrait"
        width, height = self.RESOLUTIONS[resolution]
        frame_rate = float(max(1, min(120, int(frame_rate))))
        target_frames = duration_seconds * int(frame_rate)
        frame_count = int(round((target_frames - 1) / 8.0) * 8) + 1
        if cached_prompt and use_edited_prompt and not refresh_prompt:
            return {"ui": {"optimized_prompt": [cached_prompt]}, "result": (cached_prompt, width, height, frame_count, frame_rate)}

        image_detail, temperature, top_p, max_tokens, timeout = "high", 0.35, 0.9, 1800, 120
        model_name, clear_context, purge_vram = "ComfyUI-llama-cpp", True, True
        connector = LocalLlamaCppConnector(llama_model, model_name=model_name, clear_context=clear_context, timeout=timeout)
        reference_slots = (
            reference_image_1, reference_image_2, reference_image_3, reference_image_4, reference_image_5
        )
        slot_indices = []
        image_urls = []
        encoded_sizes = []
        for slot_index, image_batch in enumerate(reference_slots, start=1):
            if image_batch is None:
                continue
            max_reference_size = 768 if slot_index == 5 else 640
            image_url = _ltx_reference_to_data_url(image_batch, max_reference_size)
            if image_url:
                slot_indices.append(slot_index)
                image_urls.append(image_url)
                encoded_sizes.append(f"R{slot_index}<={max_reference_size}px")
        if encoded_sizes:
            print("[WenWu LTX-2.3] Vision thumbnails: " + ", ".join(encoded_sizes))
        user_text = f"Video theme: {video_theme.strip()}\nDuration: {int(duration_seconds)} seconds. Build a visual plan whose number of actions, dialogue length, camera movement, and prompt length match this exact duration. Preserve all supplied dialogue exactly as written and in its original language."
        result = ""
        if not image_urls:
            result = self._chat(connector, LTX23_SYSTEM_PROMPT, user_text, [],
                                json_mode=False, image_detail=image_detail,
                                temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            result = (result or "").strip()
        if image_urls:
            attachment_map = ", ".join(
                f"attached image {attachment_index}=Reference image {slot_index}"
                for attachment_index, slot_index in enumerate(slot_indices, start=1)
            )
            fast_text = (
                user_text
                + f"\n\nThe attachment mapping is exact: {attachment_map}. "
                "Analyze all attached images together in this single pass and output the complete final LTX-2.3 prompt now. "
                "References 1-4 are Licon subject/reference slots. Reference 5, when supplied, is the fixed Licon background "
                "slot and must be the exclusive story environment unless the user explicitly requests another setting. "
                "Discard every incidental background visible in References 1-4. Keep different subjects completely separate, "
                "preserve their distinctive identities, and never blend faces, hair, bodies, clothing, accessories, voices, or roles. "
                "Follow the required two-group output structure, keep each supplied reference description near 30 English words, "
                "and do not output role markers, attachment labels, headings, numbering, analysis, or explanations."
            )
            try:
                result = self._chat(
                    connector,
                    LTX23_SYSTEM_PROMPT,
                    fast_text,
                    image_urls,
                    json_mode=False,
                    image_detail=image_detail,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                result = (result or "").strip()
                if len(result) < 80:
                    print("[WenWu LTX-2.3] Fast multi-image response was too short; using sequential fallback.")
                    result = ""
                else:
                    print(f"[WenWu LTX-2.3] Fast multi-image path succeeded with {len(image_urls)} image(s).")
            except Exception as fast_error:
                print(
                    f"[WenWu LTX-2.3] Fast multi-image path failed ({type(fast_error).__name__}: {fast_error}); "
                    "using sequential fallback."
                )
                result = ""
        if image_urls and not result:
            descriptions = []
            for index, image_url in zip(slot_indices, image_urls):
                image_prompt = (
                    f"User request: {video_theme.strip()}\nYou are analyzing Reference image {index}. Silently infer its intended role. "
                    + ("This is the fixed Licon background slot. You must classify it as [ROLE: ENVIRONMENT] and describe its usable "
                       "setting; do not reinterpret it as a subject reference. " if index == 5 else
                       "This is a Licon subject/reference slot. Classify its intended principal entity as [ROLE: SUBJECT] unless the "
                       "user explicitly assigns this numbered image as an environment. Its visible surroundings are incidental. ")
                    +
                    "Return an internal role marker followed by the description: write exactly '[ROLE: ENVIRONMENT] ' for a dedicated "
                    "setting reference or '[ROLE: SUBJECT] ' for a subject reference. If the user explicitly identifies this numbered "
                    "image as the background, scene, environment, location, room, street, landscape, or equivalent, or if it is clearly "
                    "an environment-dominant wide scene with no intended principal subject, describe only the usable "
                    "environment in approximately 30 English words and omit incidental people. Otherwise treat it as a subject "
                    "reference: describe only the intended person, character, animal, product, object, prop, vehicle, or garment "
                    "in approximately 30 English words and, unless the user explicitly asks to retain or use that specific original "
                    "background or a clearly named element from it, completely ignore its room, street, landscape, architecture, scenery, "
                    "furniture, environmental lighting, and unrelated background objects. For a person, prioritize three to five "
                    "identity-discriminating anchors such as face geometry, brows, eyes, nose, lips, skin tone, exact hairstyle, "
                    "headwear, facial markings, garment cut, palette, and accessories. For a non-person, prioritize unique silhouette, "
                    "structure, markings, materials, parts, colors, and state. A subject portrait remains ROLE: SUBJECT even when a "
                    "detailed landscape, room, street, or plantation is visible behind it. Do not invent anything. Output the role "
                    "marker and one paragraph only, with no heading, numbering, or bullet."
                )
                description = self._chat(
                    connector,
                    "You are a precise visual reference analyst. Output one English paragraph only.",
                    image_prompt,
                    [image_url],
                    json_mode=False,
                    image_detail=image_detail,
                    temperature=0.2,
                    top_p=0.85,
                    max_tokens=96,
                )
                description = (description or "").strip()
                if description:
                    descriptions.append(f"Reference image {index}: {description}")
            if descriptions:
                grounded_text = (
                    user_text
                    + "\n\nVerified reference descriptions in input order:\n"
                    + "\n".join(descriptions)
                    + "\n\nUnless the user explicitly identifies references as the same subject, keep every referenced subject "
                    "independent. Contrast their unique anchors, repeat the correct anchors in every applicable shot, and never "
                    "blend or transfer faces, hair, markings, headwear, clothing, accessories, bodies, voices, or dialogue. Treat "
                    "the supplied [ROLE: ...] markers as binding internal metadata and never print them in the final answer. When the "
                    "user says only '背景图', '背景图片', '场景图', '环境图', 'background image', or an equivalent without a number, "
                    "select the unique [ROLE: ENVIRONMENT] reference as the exclusive story setting. It overrides every setting "
                    "visible in all [ROLE: SUBJECT] references. State that environment concretely throughout the story and do not "
                    "mention, inherit, or recreate any subject-reference background. Reference image 5 is the fixed Licon background "
                    "slot: when present, its [ROLE: ENVIRONMENT] description is authoritative and exclusive unless the user explicitly "
                    "requests a different setting or location change."
                )
                result = self._chat(
                    connector,
                    LTX23_SYSTEM_PROMPT,
                    grounded_text,
                    [],
                    json_mode=False,
                    image_detail=image_detail,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                result = (result or "").strip()
        if not result:
            raise RuntimeError("LTX-2.3 prompt generation returned no text after sequential analysis of up to five reference images. Check that the GGUF model and mmproj belong to the same supported vision model package.")
        if purge_vram:
            _ltx_purge_vram(connector)
        return {"ui": {"optimized_prompt": [result]}, "result": (result, width, height, frame_count, frame_rate)}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hashlib.md5(repr(sorted(kwargs.items())).encode("utf-8")).hexdigest()
