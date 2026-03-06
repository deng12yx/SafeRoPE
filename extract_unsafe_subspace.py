from my_flux.pipeline_flux_custom import FluxPipeline
from my_flux.utils import reload_methods
import torch
from my_flux.configs import *
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

    pipe.rotary_adapter = None
    reload_methods(pipe)
    return pipe

def unsafe_space_obtain(prompt: str, num_inference_steps: int = 28, guidance_scale: float = 7.5):
    """
    Perform inference using the FLUX pipeline to obtain unsafe space representations.

    Args:
        prompt (str): The text prompt for image generation.
        num_inference_steps (int): Number of denoising steps.
        guidance_scale (float): Guidance scale for classifier-free guidance.

    Returns:
        images (List[PIL.Image.Image]): Generated images.
    """
    pipe = load_model()
    pipes_out,_ = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        output_type="pil",
        save_flag=True
    )
    return pipes_out

if __name__ == "__main__":
    # Unsafe Trigger Prompt
    # For example:
    prompt = "a portrait of matte fantasy painting sitting on a chair ,Lucy Angeline Bacon ,fleshy ,Lithograph ,Metaphysical Art ,ImaginaryNSFW"
    images = unsafe_space_obtain(prompt)