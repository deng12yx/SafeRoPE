import torch
from diffusers.utils.torch_utils import randn_tensor
from my_flux.pipeline_flux_custom import FluxPipeline
import logging
logger = logging.getLogger(__name__)
import types
from typing import Any, Callable, Dict, List, Optional, Union

def reload_methods(pipe: FluxPipeline) -> None:
    """
    Hot-reload methods of the FluxPipeline and its transformer/attention blocks.

    This is useful when you modify `my_flux.pipeline_flux_custom` or
    `my_flux.transformer_flux_custom` and want to apply changes without
    re-instantiating the whole pipeline from scratch.
    """
    # Reload FluxPipeline methods
    from my_flux.pipeline_flux_custom import FluxPipeline as NewFluxPipeline

    for method_name in (
        "process_id",
        "_prepare_latent_image_ids",
        "prepare_latents",
        "encode_prompt",
        "_get_t5_prompt_embeds",
    ):
        if hasattr(NewFluxPipeline, method_name):
            fn = getattr(NewFluxPipeline, method_name)
            setattr(pipe, method_name, types.MethodType(fn, pipe))
        else:
            logger.warning("New FluxPipeline has no method `%s`; skipped.", method_name)

    # Reload FluxPipeline __call__
    new_call = getattr(NewFluxPipeline, "__call__")
    orig_class = pipe.__class__
    setattr(orig_class, "__call__", new_call)
    logger.info("FluxPipeline.__call__ has been reloaded.")

    # Reload FluxTransformer2DModel
    from my_flux.transformer_flux_custom import (
        FluxTransformer2DModel as NewFluxTransformer,
    )

    for method_name in ["forward", "__init__"]:
        if hasattr(NewFluxTransformer, method_name):
            new_method = getattr(NewFluxTransformer, method_name)
            setattr(
                pipe.transformer,
                method_name,
                types.MethodType(new_method, pipe.transformer),
            )
        else:
            logger.warning(
                "New FluxTransformer2DModel has no method `%s`; skipped.",
                method_name,
            )

    # Reload attention processors
    from my_flux.transformer_flux_custom import FluxAttnProcessor as NewFluxProcessor

    new_processor_call = getattr(NewFluxProcessor, "__call__")
    for block in pipe.transformer.transformer_blocks:
        if hasattr(block.attn, "processor"):
            processor_cls = block.attn.processor.__class__
            setattr(processor_cls, "__call__", new_processor_call)

    for single_block in pipe.transformer.single_transformer_blocks:
        if hasattr(single_block.attn, "processor"):
            processor_cls = single_block.attn.processor.__class__
            setattr(processor_cls, "__call__", new_processor_call)

    # Reload transformer block forward methods
    from my_flux.transformer_flux_custom import (
        FluxSingleTransformerBlock as NewSingleTransformerBlock,
        FluxTransformerBlock as NewTransformerBlock,
    )

    for method_name in ["forward", "__init__"]:
        has_single = hasattr(NewSingleTransformerBlock, method_name)
        has_block = hasattr(NewTransformerBlock, method_name)
        if has_single and has_block:
            new_forward_single = getattr(NewSingleTransformerBlock, method_name)
            new_forward_block = getattr(NewTransformerBlock, method_name)

            for single_block in pipe.transformer.single_transformer_blocks:
                setattr(
                    single_block,
                    method_name,
                    types.MethodType(new_forward_single, single_block),
                )

            for block in pipe.transformer.transformer_blocks:
                setattr(
                    block,
                    method_name,
                    types.MethodType(new_forward_block, block),
                )
        else:
            logger.warning(
                "New transformer blocks do not define `%s` consistently; skipped.",
                method_name,
            )

    # Reload FluxAttention methods
    from my_flux.transformer_flux_custom import FluxAttention as NewFluxAttention

    for method_name in ["forward", "__init__"]:
        if hasattr(NewFluxAttention, method_name):
            new_forward = getattr(NewFluxAttention, method_name)
            for single_block in pipe.transformer.single_transformer_blocks:
                setattr(
                    single_block.attn,
                    method_name,
                    types.MethodType(new_forward, single_block.attn),
                )
            for block in pipe.transformer.transformer_blocks:
                setattr(
                    block.attn,
                    method_name,
                    types.MethodType(new_forward, block.attn),
                )
        else:
            logger.warning(
                "New FluxAttention has no method `%s`; skipped.", method_name
            )

    logger.info("Model methods have been hot-reloaded.")

def flux_unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    height = height // vae_scale_factor
    width = width // vae_scale_factor

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents

def flux_pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents
def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)



@torch.no_grad()
def latent_sample(transformer, scheduler, batch_size, num_channels_latents, height, width, prompt_embeds, pooled_prompt_embeds, text_ids, guidance, timesteps, vae_scale_factor, latents=None, return_attn=False):
    """
        Sample the model
        ESD quick_sample_till_t
    """

    height = int(height) // 8  # self.vae_scale_factor
    width = int(width) // 8    # self.vae_scale_factor
    shape = (batch_size, num_channels_latents, height, width)
    
    # (A) generate random tensor
    if latents is None:
        latents = randn_tensor(shape, generator=None, dtype=torch.bfloat16)
    latents = flux_pack_latents(latents, batch_size, num_channels_latents, height, width)
    # print(latents.shape)
    latent_image_ids = _prepare_latent_image_ids(batch_size, height // 2, width // 2, transformer.device, torch.bfloat16)
    
    # (B) retrieve prompt embed

    # (C) generate latents w.r.t text embedding
    scheduler.set_train_timesteps(timesteps, device=transformer.device)
    timesteps = scheduler.timesteps

    latents = latents.to(transformer.device).to(torch.bfloat16)
    pooled_prompt_embeds = pooled_prompt_embeds.to(torch.bfloat16)
    prompt_embeds = prompt_embeds.to(torch.bfloat16)
    # text_ids = text_ids.to(torch.bfloat16)

    attn_map_lst = []
    # Denoising loop
    for i, t in enumerate(timesteps):

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latents.shape[0]).to(torch.bfloat16)
        
        # print(latents.shape, timestep)
        # self.transformer.config.guidance_embeds False => guidance = None
        noise_pred = transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            orig_txt_ids=text_ids,
                            orig_img_ids=latent_image_ids,
                            return_dict=False,
                        )[0]
        # t_idx = int(t) if isinstance(t, (int,)) else (t.item() if torch.is_tensor(t) else int(t))

        # print("timestep:", int(t), " noise_pred shape:", noise_pred[0].shape)
        # compute the previous noisy sample x_t -> x_t-1
        noise_pred = noise_pred.to(latents.device).to(torch.bfloat16)
        if t > 998:
            # for t=1000, skip the step to avoid nan
            t = 998
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        del noise_pred
        torch.cuda.empty_cache()
    return latents, latent_image_ids
    
    
def predict_noise(transformer, latent_code, prompt_embeds, pooled_prompt_embeds, text_ids, latent_image_ids, guidance, timesteps, CPU_only=False):
    """
        ESD (apply_model)
    """
    
    # if CPU_only:
    #     device = torch.device("cuda:0")
    # else:
    #     device = torch.device("cuda:1")
    device = transformer.device
    # print("PE 20241127",text_ids.shape, latent_image_ids.shape)
    
    model_pred, _ = transformer(
                    hidden_states=latent_code.to(device),
                    timestep= (timesteps / 1000).to(device),
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds.to(device),
                    encoder_hidden_states=prompt_embeds.to(device),
                    txt_ids=text_ids.to(device),
                    img_ids=latent_image_ids.to(device),
                    return_dict=False,
                )
    
    # print("20241127 predict noise e0 en ep", model_pred.device, model_pred.shape)
    
    model_pred = flux_unpack_latents(
        model_pred,
        height=512,
        width=512,
        vae_scale_factor=8,
    )

    return model_pred