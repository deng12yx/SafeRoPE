"""
Training script for SafeRoPE rotary adapter on FLUX-based FluxPipeline.

This script:
- Loads a custom FLUX pipeline.
- Builds COCO and unsafe prompt datasets.
- Trains a rotary adapter with safe and unsafe losses.
"""
import logging
import math
import os
import copy
import gc
from typing import Iterator, Tuple

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from diffusers.optimization import get_scheduler

from my_flux.pipeline_flux_custom import FluxPipeline
from my_flux.schedule_custom_process import CustomFlowMatchEulerDiscreteScheduler
from my_flux.transformer_flux_custom import RotaryAdapter
from my_flux.process_dataset import (
    CocoDataset,
    UnsafePromptDataset,
    SafeUnlearningDataset, 
)
from my_flux.utils import latent_sample, predict_noise,reload_methods
from my_flux.configs import *

import types

#####################################################
# Logging and global setup

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)

os.makedirs(save_model_path, exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(save_model_path, "tb_logs"))

# Global loss containers (optional, mainly for debugging / plotting)
safe_losses = []
unsafe_losses = []
all_losses = []


def endless_iterator(loader):
    while True:
        for batch in loader:
            yield batch


def endless_iterator(loader: DataLoader) -> Iterator:
    """
    Wrap a DataLoader into an endless iterator.

    This avoids manual epoch restarts and allows simple `next(it)` usage.
    """
    while True:
        for batch in loader:
            yield batch



def process_dataset() -> Tuple[DataLoader, DataLoader]:
    """
    Build COCO and unsafe prompt datasets and return their DataLoaders.

    Returns:
        train_loader_coco (DataLoader): Loader over COCO image-text pairs.
        train_loader_unsafe (DataLoader): Loader over unsafe prompt text dataset.
    """
    image_transforms = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            # Normalize to [-1, 1] for 3-channel images
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    coco_dataset = CocoDataset(
        image_dir=COCO_IMG_PATH,
        caption_path=COCO_CAPTION_PATH,
        transform=image_transforms,
    )

    unsafe_dataset = UnsafePromptDataset(txt_path=UNSAFE_PROMPT_PATH)

    train_loader_coco = DataLoader(
        coco_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    train_loader_unsafe = DataLoader(
        unsafe_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader_coco, train_loader_unsafe

def train_rotary(
    args,  # kept for potential future extension
    pipe: FluxPipeline,
    train_loader_coco: Iterator,
    train_loader_unsafe: Iterator,
    height: int = 1024,
    width: int = 1024,
    total_len: int = 0,
) -> None:
    """
    Train the rotary adapter in a SafeRoPE-style setting.

    This function:
    - Freezes all model parameters except the rotary adapter.
    - Alternates between safe and unsafe training steps.
    - Uses flow-matching for safe data and contrastive behavior for unsafe data.
    """

    # ------------------------------------------------------------------
    # Freeze all modules except the rotary adapter
    # ------------------------------------------------------------------
    for name, module in pipe.__dict__.items():
        if isinstance(module, torch.nn.Module):
            for param in module.parameters():
                param.requires_grad = False

    for param in pipe.rotary_adapter.parameters():
        param.requires_grad = True

    logger.info("All parameters except rotary adapter have been frozen.")

    # ------------------------------------------------------------------
    # Optimizer and scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        pipe.rotary_adapter.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        name="constant",  # constant learning rate
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=max_train_steps,
        num_cycles=lr_num_cycles,
        power=lr_power,
    )

    height = height or pipe.default_sample_size * pipe.vae_scale_factor
    width = width or pipe.default_sample_size * pipe.vae_scale_factor

    # ------------------------------------------------------------------
    # Generation / sampling configuration
    # ------------------------------------------------------------------
    max_sequence_length = 512
    num_images_per_prompt = 1
    guidance_scale = 7.5

    device = pipe._execution_device
    weight_dtype = torch.float32
    rng = torch.Generator(device=device)

    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    ddim_steps = 28

    noise_scheduler = CustomFlowMatchEulerDiscreteScheduler.from_pretrained(
        model_path,
        subfolder="scheduler",
    )

    # in_channels is for packed latent channels; VAE in_channels=4 -> transformer in_channels=16
    num_channels_latents = pipe.transformer.config.in_channels // 4

    lora_scale = None

    vae_config_shift_factor = pipe.vae.config.shift_factor
    vae_config_scaling_factor = pipe.vae.config.scaling_factor
    vae_config_block_out_channels = pipe.vae.config.block_out_channels
    vae_scale_factor = 2 ** len(vae_config_block_out_channels)

    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # Estimate number of optimization steps per epoch
    num_update_steps_per_epoch = math.ceil(
        (safe_repeat + unsafe_repeat) / gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    criterion = torch.nn.MSELoss()

    total_batch_size = train_batch_size * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", total_len)
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Instantaneous batch size per device = %d", train_batch_size)
    logger.info(
        "  Total train batch size (parallel + accumulation) = %d", total_batch_size
    )
    logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", max_train_steps)

    global_step = 0
    first_epoch = 0
    logger.info(
        "Total training epochs: %d, total training steps: %d",
        num_train_epochs,
        max_train_steps,
    )
    logger.info("Starting training...")

    initial_global_step = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
    )

    # ------------------------------------------------------------------
    # Regularizer on S_param to constrain the rotation magnitude
    # ------------------------------------------------------------------
    def loss_S_param() -> torch.Tensor:
        """
        Regularize the skew-symmetric parameter matrices S_param_Single.

        This encourages the exponential map of the skew-symmetric matrix
        to remain close to identity, constraining the rotation strength.
        """
        loss_all = 0.0
        dim = pipe.rotary_adapter.S_param_Single[0].shape[-1]
        # Identity should be on the same device and dtype as rotation parameters
        identity = torch.eye(
            dim,
            device=pipe.rotary_adapter.S_param_Single[0].device,
            dtype=pipe.rotary_adapter.S_param_Single[0].dtype,
        )

        for param in pipe.rotary_adapter.S_param_Single:
            # Only use the first matrix in the param (assuming shape [N, d, d])
            param_scaled = param[0] * 0.01
            sym_matrix = param_scaled - param_scaled.transpose(-2, -1)
            loss = criterion(torch.matrix_exp(sym_matrix), identity)
            loss_all = loss_all + loss

        return loss_all

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(first_epoch, num_train_epochs):
        # Choose whether this epoch focuses on safe or unsafe data
        if torch.rand(1).item() < p_safe:
            current_class = "safe"
            repeat = safe_repeat
        else:
            current_class = "unsafe"
            repeat = unsafe_repeat

        for _ in range(repeat):
            # -------------- SAFE STEP --------------
            if global_step == 0 or current_class == "safe":
                batch = next(train_loader_coco)
                prompt = batch["text"]

                # 1. Sanity-check inputs
                pipe.check_inputs(
                    prompt,
                    None,
                    height,
                    width,
                    max_sequence_length=max_sequence_length,
                )

                # 2. Compute batch size from prompt type
                if isinstance(prompt, str):
                    batch_size = 1
                elif isinstance(prompt, list):
                    batch_size = len(prompt)
                else:
                    raise ValueError("`prompt` must be of type `str` or `list`.")

                # 3. Encode images to latent space
                pixel_values = batch["image"].to(dtype=pipe.vae.dtype, device=device)
                model_input = pipe.vae.encode(pixel_values).latent_dist.sample()

                model_input = (model_input - vae_config_shift_factor) * (
                    vae_config_scaling_factor
                )
                model_input = model_input.to(dtype=weight_dtype)

                latent_image_ids = pipe._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2] // 2,
                    model_input.shape[3] // 2,
                    device,
                    weight_dtype,
                )

                # 4. Sample noise and encode prompts
                noise = torch.randn_like(model_input)

                (
                    prompt_embeds,
                    pooled_embeds,
                    text_ids,
                    _,
                ) = pipe.encode_prompt(
                    prompt=prompt,
                    device=device,
                    num_images_per_prompt=1,
                    max_sequence_length=max_sequence_length,
                )

                # Map a continuous diffusion step to the discrete DDPM scheduler space
                t_enc = torch.randint(ddim_steps, (1,), device=device)
                og_num = round((int(t_enc) / ddim_steps) * 1000)
                og_num_lim = round(((int(t_enc) + 1) / ddim_steps) * 1000)
                t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=device)

                noisy_model_input = noise_scheduler_copy.add_noise(
                    model_input, noise, t_enc_ddpm
                )

                packed_noisy_model_input = pipe._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )

                if pipe.transformer.config.guidance_embeds:
                    guidance = torch.tensor([guidance_scale], device=device)
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None

                # 5. Forward through transformer with rotary adapter
                model_pred = pipe.transformer(
                    hidden_states=packed_noisy_model_input.to(
                        dtype=weight_dtype, device=device
                    ),
                    timestep=t_enc_ddpm / 1000,  # scheduler expects [0,1] range
                    guidance=(
                        guidance.to(dtype=weight_dtype, device=device)
                        if guidance is not None
                        else None
                    ),
                    pooled_projections=pooled_embeds.to(
                        dtype=weight_dtype, device=device
                    ),
                    encoder_hidden_states=prompt_embeds.to(
                        dtype=weight_dtype, device=device
                    ),
                    txt_ids=text_ids.to(dtype=weight_dtype, device=device),
                    img_ids=latent_image_ids.to(dtype=weight_dtype, device=device),
                    orig_txt_ids=text_ids.to(dtype=weight_dtype, device=device),
                    orig_img_ids=latent_image_ids.to(dtype=weight_dtype, device=device),
                    rotary_adapter=pipe.rotary_adapter,
                    return_dict=False,
                )[0]

                # Unpack latents back to image-shaped tensor for loss
                vae_scale_factor_safe = 8  # explicitly fix for flow matching loss
                model_pred = pipe._unpack_latents(
                    model_pred,
                    height=height,
                    width=width,
                    vae_scale_factor=vae_scale_factor_safe,
                )

                # 6. Flow-matching loss
                target = noise - model_input
                model_pred = model_pred.to(dtype=target.dtype, device=target.device)
                loss_safe = criterion(model_pred, target)

                loss_s_param = loss_S_param() * 100.0

                # Currently only the safe loss is used for optimization,
                # while S_param loss is tracked for monitoring.
                loss = loss_safe

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)
                global_step += 1

                logs = {
                    "loss_safe": loss_safe.item(),
                    "loss_s_param": loss_s_param.item(),
                }
                progress_bar.set_postfix(**logs)

                safe_losses.append(loss.item())
                all_losses.append(loss.item())

                # TensorBoard logging
                writer.add_scalar("loss/safe", loss_safe.item(), global_step)
                writer.add_scalar("loss/s_param", loss_s_param.item(), global_step)

            # -------------- UNSAFE STEP --------------
            else:
                batch = next(train_loader_unsafe)
                prompt = batch["text"]

                # 1. Sanity-check inputs
                pipe.check_inputs(
                    prompt,
                    None,
                    height,
                    width,
                    max_sequence_length=max_sequence_length,
                )

                # 2. Compute batch size from prompt type
                if isinstance(prompt, str):
                    batch_size = 1
                elif isinstance(prompt, list):
                    batch_size = len(prompt)
                else:
                    raise ValueError("`prompt` must be of type `str` or `list`.")

                (
                    unsafe_prompt_embeds,
                    unsafe_pooled_prompt_embeds,
                    unsafe_text_ids,
                    _,
                ) = pipe.encode_prompt(
                    prompt=prompt,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    lora_scale=lora_scale,
                )

                latents, latent_image_ids = pipe.prepare_latents(
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    weight_dtype,
                    device,
                    rng,
                )

                # Fixing timestep to 1 for unsafe optimization
                fix_timestep = torch.tensor([1], device=device)

                # Baseline prediction without rotary adapter intervention
                with torch.no_grad():
                    unsafe_model_pred = pipe.transformer(
                        hidden_states=latents.to(
                            dtype=weight_dtype, device=device
                        ),
                        timestep=fix_timestep,
                        guidance=(
                            guidance.to(dtype=weight_dtype, device=device)
                            if guidance is not None
                            else None
                        ),
                        pooled_projections=unsafe_pooled_prompt_embeds.to(
                            dtype=weight_dtype, device=device
                        ),
                        encoder_hidden_states=unsafe_prompt_embeds.to(
                            dtype=weight_dtype, device=device
                        ),
                        txt_ids=unsafe_text_ids.to(
                            dtype=weight_dtype, device=device
                        ),
                        img_ids=latent_image_ids.to(
                            dtype=weight_dtype, device=device
                        ),
                        orig_txt_ids=unsafe_text_ids.to(
                            dtype=weight_dtype, device=device
                        ),
                        orig_img_ids=latent_image_ids.to(
                            dtype=weight_dtype, device=device
                        ),
                        # rotary_adapter is intentionally omitted here
                        return_dict=False,
                    )[0]

                    unsafe_model_pred = pipe._unpack_latents(
                        unsafe_model_pred,
                        height=height,
                        width=width,
                        vae_scale_factor=vae_scale_factor,
                    )

                # Prediction with rotary adapter
                model_pred = pipe.transformer(
                    hidden_states=latents.to(dtype=weight_dtype, device=device),
                    timestep=fix_timestep,
                    guidance=(
                        guidance.to(dtype=weight_dtype, device=device)
                        if guidance is not None
                        else None
                    ),
                    pooled_projections=unsafe_pooled_prompt_embeds.to(
                        dtype=weight_dtype, device=device
                    ),
                    encoder_hidden_states=unsafe_prompt_embeds.to(
                        dtype=weight_dtype, device=device
                    ),
                    txt_ids=unsafe_text_ids.to(dtype=weight_dtype, device=device),
                    img_ids=latent_image_ids.to(dtype=weight_dtype, device=device),
                    orig_txt_ids=unsafe_text_ids.to(dtype=weight_dtype, device=device),
                    orig_img_ids=latent_image_ids.to(dtype=weight_dtype, device=device),
                    rotary_adapter=pipe.rotary_adapter,
                    return_dict=False,
                )[0]

                model_pred = pipe._unpack_latents(
                    model_pred,
                    height=height,
                    width=width,
                    vae_scale_factor=vae_scale_factor,
                )

                # 3. Unsafe loss: encourage deviation from unsafe baseline
                loss_unsafe = -criterion(unsafe_model_pred, model_pred)
                loss_unsafe = torch.tanh(loss_unsafe)

                loss_s_param = loss_S_param() * 100.0

                logs = {
                    "loss_unsafe": loss_unsafe.item(),
                    "loss_s_param": loss_s_param.item(),
                }
                unsafe_losses.append(loss_unsafe.item())
                all_losses.append(loss_unsafe.item())
                progress_bar.set_postfix(**logs)

                optimizer.zero_grad()
                loss = loss_unsafe
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)
                global_step += 1

                # TensorBoard logging
                writer.add_scalar("loss/unsafe", loss_unsafe.item(), global_step)
                writer.add_scalar("loss/s_param", loss_s_param.item(), global_step)

        # Periodic checkpointing
        if (epoch + 1) % 10 == 0:
            os.makedirs(save_model_path, exist_ok=True)
            save_path = os.path.join(
                save_model_path, f"rotary_adapter_s_param_epoch{epoch + 1}.pth"
            )
            torch.save(pipe.rotary_adapter.state_dict(), save_path)
            logger.info("Saved rotary adapter checkpoint to %s", save_path)

    # Final checkpoint
    os.makedirs(save_model_path, exist_ok=True)
    final_save_path = os.path.join(save_model_path, "rotary_adapter_s_param.pth")
    torch.save(pipe.rotary_adapter.state_dict(), final_save_path)
    logger.info("Final rotary adapter S_param model saved to %s", final_save_path)


def load_model() -> FluxPipeline:
    """
    Load the base FLUX pipeline and attach the RotaryAdapter.

    Returns:
        pipe (FluxPipeline): Loaded and method-reloaded pipeline with rotary adapter.
    """
    from my_flux.transformer_flux_custom import RotaryAdapter, FluxTransformer2DModel

    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="balanced",  # let accelerate distribute submodules across GPUs
    )

    # Replace transformer with a more explicit loading of FluxTransformer2DModel
    transformer_before = pipe.transformer
    pipe.transformer = FluxTransformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    del transformer_before
    gc.collect()
    torch.cuda.empty_cache()

    pipe.rotary_adapter = RotaryAdapter(
        pipe.transformer.config,
        r=20,
        dtype=torch.float32,
    )

    # Optionally load a pretrained rotary adapter checkpoint here, e.g.:
    # pipe.rotary_adapter.load_state_dict(
    #     torch.load("/save_s_param_model/rotary_adapter_s_param_epoch10.pth")
    # )

    reload_methods(pipe)
    return pipe


#################################################
def main() -> None:
    pipe = load_model()
    train_loader_coco, train_loader_unsafe = process_dataset()
    total_len = len(train_loader_coco) + len(train_loader_unsafe)
    logger.info("Total length of training set (approx): %d", total_len)

    it_safe = endless_iterator(train_loader_coco)
    it_unsafe = endless_iterator(train_loader_unsafe)

    train_rotary(
        args=None,
        pipe=pipe,
        train_loader_coco=it_safe,
        train_loader_unsafe=it_unsafe,
        height=1024,
        width=1024,
        total_len=total_len,
    )


if __name__ == "__main__":
    main()