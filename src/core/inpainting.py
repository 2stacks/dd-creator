import torch
import gc
import numpy as np
from PIL import Image, ImageOps


class LamaInpainter:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def load_model(self):
        if self.model is not None:
            return
        print("Loading LaMa inpainting model...")
        from simple_lama_inpainting import SimpleLama
        self.model = SimpleLama()
        print("LaMa model loaded.")

    def inpaint(self, image_path, mask):
        """Inpaint image using LaMa.

        Args:
            image_path: Path to source image
            mask: PIL Image (mode 'L'), white=inpaint region

        Returns:
            PIL Image with inpainted result
        """
        self.load_model()

        image = Image.open(image_path).convert("RGB")
        image = ImageOps.exif_transpose(image)

        # Ensure mask matches image size
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)

        # simple-lama expects RGB image and L mask
        if mask.mode != "L":
            mask = mask.convert("L")

        result = self.model(image, mask)
        return result

    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("LaMa model unloaded.")


class SDInpainter:
    # Available models
    MODELS = {
        "SD 1.5 Inpainting (~6GB)": "stable-diffusion-v1-5/stable-diffusion-inpainting",
        "SDXL Inpainting (~10GB)": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    }

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self._loaded_model_id = None

    def load_model(self, model_id):
        if self.pipe is not None and self._loaded_model_id == model_id:
            return

        # Unload existing if different model
        if self.pipe is not None:
            self.unload_model()

        print(f"Loading SD inpainting model: {model_id}...")

        is_xl = "xl" in model_id.lower()

        if is_xl:
            from diffusers import StableDiffusionXLInpaintPipeline
            self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                variant="fp16",
            ).to(self.device)
        else:
            from diffusers import StableDiffusionInpaintPipeline
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                variant="fp16",
            ).to(self.device)

        self.pipe.enable_attention_slicing()
        self._loaded_model_id = model_id
        print(f"SD inpainting model loaded: {model_id}")

    def inpaint(self, image_path, mask, prompt="", negative_prompt="",
                num_inference_steps=30, guidance_scale=7.5, strength=0.85):
        """Inpaint image using Stable Diffusion.

        Args:
            image_path: Path to source image
            mask: PIL Image (mode 'L'), white=inpaint region
            prompt: Text prompt for inpainting
            negative_prompt: Negative prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            strength: How much to transform the masked area (0-1)

        Returns:
            PIL Image with inpainted result at original resolution
        """
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        image = Image.open(image_path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        original_size = image.size  # (width, height)

        # Ensure mask matches image size
        if mask.mode != "L":
            mask = mask.convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)

        # Determine inference resolution
        is_xl = "xl" in self._loaded_model_id.lower()
        target_size = 1024 if is_xl else 512

        # Resize for inference
        image_resized = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        mask_resized = mask.resize((target_size, target_size), Image.Resampling.NEAREST)

        # Convert mask to RGB for pipeline
        mask_rgb = mask_resized.convert("RGB")

        result = self.pipe(
            prompt=prompt or "",
            negative_prompt=negative_prompt or "",
            image=image_resized,
            mask_image=mask_rgb,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            width=target_size,
            height=target_size,
        ).images[0]

        # Resize result back to original resolution
        result_full = result.resize(original_size, Image.Resampling.LANCZOS)

        # Composite: use inpainted result only in masked area, original elsewhere
        mask_full = mask.point(lambda x: 255 if x > 128 else 0)
        composite = Image.composite(result_full, image, mask_full)

        return composite

    def unload_model(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self._loaded_model_id = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("SD inpainting model unloaded.")


# Global instance patterns
_global_lama = None
_global_sd = None


def get_lama_inpainter():
    global _global_lama
    if _global_lama is None:
        _global_lama = LamaInpainter()
    return _global_lama


def get_sd_inpainter():
    global _global_sd
    if _global_sd is None:
        _global_sd = SDInpainter()
    return _global_sd
