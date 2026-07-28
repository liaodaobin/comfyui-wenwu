import inspect
import logging

import torch

import comfy.conds
import comfy.model_management
from comfy.ldm.wan.model import AudioInjector_WAN, WanModel_S2V
from comfy.model_base import WAN22_S2V

# WenWu carries this compatibility patch so Bernini-R-S2V reference
# conditioning works without any third-party custom node.  Every patch below
# is guarded by marker attributes, making this safe when another package has
# already applied an equivalent patch.


def _append_context_latents(self, x, kwargs):
    context_latents = kwargs.get("context_latents", None)
    if context_latents is None:
        return x
    for lat in context_latents:
        cl = self.patch_embedding(lat.float().to(x.device)).to(x.dtype).flatten(2).transpose(1, 2)
        x = torch.cat([x, cl], dim=1)
    return x


def _patch_wan_model_s2v_forward():
    if getattr(WanModel_S2V.forward_orig, "__wan_bernini_s2v_v2_patch__", False):
        return

    try:
        source = inspect.getsource(WanModel_S2V.forward_orig)
    except (OSError, TypeError):
        source = ""
    if "context_latents" in source and getattr(WanModel_S2V.forward_orig, "__wan_bernini_s2v_patch__", False):
        WanModel_S2V.forward_orig.__wan_bernini_s2v_v2_patch__ = True
        return

    original = WanModel_S2V.forward_orig

    def forward_orig(
        self,
        x,
        t,
        context,
        audio_embed=None,
        reference_latent=None,
        control_video=None,
        reference_motion=None,
        clip_fea=None,
        freqs=None,
        transformer_options={},
        **kwargs,
    ):
        if audio_embed is not None:
            num_embeds = x.shape[-3] * 4
            audio_emb_global, audio_emb = self.casual_audio_encoder(audio_embed[:, :, :, :num_embeds])
        else:
            audio_emb = None
            audio_emb_global = None

        bs, _, time, height, width = x.shape
        x = self.patch_embedding(x.float()).to(x.dtype)
        if control_video is not None:
            x = x + self.cond_encoder(control_video)

        if t.ndim == 1:
            t = t.unsqueeze(1).repeat(1, x.shape[2])

        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        seq_len = x.size(1)

        cond_mask_weight = comfy.model_management.cast_to(self.trainable_cond_mask.weight, dtype=x.dtype, device=x.device).unsqueeze(1).unsqueeze(1)
        x = x + cond_mask_weight[0]
        x = _append_context_latents(self, x, kwargs)

        if reference_latent is not None:
            ref = self.patch_embedding(reference_latent.float()).to(x.dtype)
            ref = ref.flatten(2).transpose(1, 2)
            freqs_ref = self.rope_encode(reference_latent.shape[-3], reference_latent.shape[-2], reference_latent.shape[-1], t_start=max(30, time + 9), device=x.device, dtype=x.dtype)
            ref = ref + cond_mask_weight[1]
            x = torch.cat([x, ref], dim=1)
            freqs = torch.cat([freqs, freqs_ref], dim=1)
            t = torch.cat([t, torch.zeros((t.shape[0], reference_latent.shape[-3]), device=t.device, dtype=t.dtype)], dim=1)

        if reference_motion is not None:
            motion_encoded, freqs_motion = self.frame_packer(reference_motion, self)
            motion_encoded = motion_encoded + cond_mask_weight[2]
            x = torch.cat([x, motion_encoded], dim=1)
            freqs = torch.cat([freqs, freqs_motion], dim=1)
            t = torch.repeat_interleave(t, 2, dim=1)
            t = torch.cat([t, torch.zeros((t.shape[0], 3), device=t.device, dtype=t.dtype)], dim=1)

        from comfy.ldm.wan.model import sinusoidal_embedding_1d

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
        e = e.reshape(t.shape[0], -1, e.shape[-1])
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        context = self.text_embedding(context)

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        transformer_options["total_blocks"] = len(self.blocks)
        transformer_options["block_type"] = "double"
        for i, block in enumerate(self.blocks):
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], transformer_options=args["transformer_options"])
                    return out
                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs, "transformer_options": transformer_options}, {"original_block": block_wrap})
                x = out["img"]
            else:
                x = block(x, e=e0, freqs=freqs, context=context, transformer_options=transformer_options)
            if audio_emb is not None:
                inject_scale = kwargs.get("audio_inject_scale", 1.0)
                if isinstance(inject_scale, torch.Tensor):
                    inject_scale = inject_scale.reshape(-1)[0].item()
                x = self.audio_injector(
                    x, i, audio_emb, audio_emb_global, seq_len,
                    scale=inject_scale,
                    token_mask=kwargs.get("audio_inject_mask", None),
                )
        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return x

    forward_orig.__wan_bernini_s2v_v2_patch__ = True
    forward_orig.__wan_bernini_s2v_patch__ = True
    forward_orig.__wan_bernini_s2v_original__ = original
    WanModel_S2V.forward_orig = forward_orig


def _patch_audio_injector():
    if getattr(AudioInjector_WAN.forward, "__wan_bernini_s2v_v2_masked_patch__", False):
        return

    original_forward = AudioInjector_WAN.forward

    def forward(self, x, block_id, audio_emb, audio_emb_global, seq_len, scale=1.0, token_mask=None):
        if token_mask is None:
            return original_forward(self, x, block_id, audio_emb, audio_emb_global, seq_len, scale=scale)

        audio_attn_id = self.injected_block_id.get(block_id, None)
        if audio_attn_id is None:
            return x

        from einops import rearrange

        num_frames = audio_emb.shape[1]
        input_hidden_states = rearrange(x[:, :seq_len], "b (t n) c -> (b t) n c", t=num_frames)
        if self.enable_adain and self.adain_mode == "attn_norm":
            audio_emb_global = rearrange(audio_emb_global, "b t n c -> (b t) n c")
            adain_hidden_states = self.injector_adain_layers[audio_attn_id](input_hidden_states, temb=audio_emb_global[:, 0])
            attn_hidden_states = adain_hidden_states
        else:
            attn_hidden_states = self.injector_pre_norm_feat[audio_attn_id](input_hidden_states)

        if audio_emb.dim() == 3:
            attn_audio_emb = rearrange(audio_emb, "b t c -> (b t) 1 c", t=num_frames)
        else:
            attn_audio_emb = rearrange(audio_emb, "b t n c -> (b t) n c", t=num_frames)

        residual_out = self.injector[audio_attn_id](x=attn_hidden_states, context=attn_audio_emb)
        residual_out = rearrange(residual_out, "(b t) n c -> b (t n) c", t=num_frames)

        if token_mask.ndim == 4:
            token_mask = token_mask.flatten(1, 2)
        if token_mask.shape[1] == residual_out.shape[1]:
            residual_out = residual_out * token_mask.to(device=residual_out.device, dtype=residual_out.dtype)
        else:
            logging.warning(
                "comfyui-wenwu: mask length %s does not match token count %s; using global audio injection",
                token_mask.shape[1],
                residual_out.shape[1],
            )

        x[:, :seq_len] = x[:, :seq_len] + residual_out * scale
        return x

    forward.__wan_bernini_s2v_v2_masked_patch__ = True
    forward.__wan_bernini_s2v_masked_patch__ = True
    forward.__wan_bernini_s2v_masked_original__ = original_forward
    AudioInjector_WAN.forward = forward


def _patch_wan22_s2v_extra_conds():
    if getattr(WAN22_S2V.extra_conds, "__wan_bernini_s2v_v2_masked_patch__", False):
        return

    original_extra_conds = WAN22_S2V.extra_conds

    def extra_conds(self, **kwargs):
        out = original_extra_conds(self, **kwargs)
        audio_inject_mask = kwargs.get("audio_inject_mask", None)
        if audio_inject_mask is not None:
            out["audio_inject_mask"] = comfy.conds.CONDRegular(audio_inject_mask)
        audio_inject_scale = kwargs.get("audio_inject_scale", None)
        if audio_inject_scale is not None:
            out["audio_inject_scale"] = comfy.conds.CONDRegular(torch.FloatTensor([audio_inject_scale]))
        return out

    extra_conds.__wan_bernini_s2v_v2_masked_patch__ = True
    extra_conds.__wan_bernini_s2v_masked_patch__ = True
    extra_conds.__wan_bernini_s2v_masked_original__ = original_extra_conds
    WAN22_S2V.extra_conds = extra_conds

    original_resize = WAN22_S2V.resize_cond_for_context_window

    def resize_cond_for_context_window(self, cond_key, cond_value, window, x_in, device, retain_index_list=[]):
        if cond_key == "audio_inject_mask":
            mask = cond_value.cond
            if mask.ndim == 4 and mask.shape[1] == x_in.shape[2]:
                return cond_value._copy_with(window.get_tensor(mask, device, dim=1))
        return original_resize(self, cond_key, cond_value, window, x_in, device, retain_index_list=retain_index_list)

    resize_cond_for_context_window.__wan_bernini_s2v_v2_masked_patch__ = True
    resize_cond_for_context_window.__wan_bernini_s2v_masked_patch__ = True
    resize_cond_for_context_window.__wan_bernini_s2v_masked_original__ = original_resize
    WAN22_S2V.resize_cond_for_context_window = resize_cond_for_context_window


def apply_model_patches():
    _patch_wan_model_s2v_forward()
    _patch_audio_injector()
    _patch_wan22_s2v_extra_conds()
    logging.info("comfyui-wenwu: applied built-in Bernini S2V model patches")
