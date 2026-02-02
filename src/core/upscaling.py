import torch
import gc
import os
from PIL import Image, ImageOps

# Spandrel is a universal loader for ESRGAN/Real-ESRGAN/SwinIR models
import spandrel

# Directory where user places .pth model files
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".models")


def get_available_models() -> list[str]:
    """Scan .models/ directory for available upscaler model files."""
    if not os.path.isdir(MODELS_DIR):
        return []

    valid_extensions = ('.pth', '.safetensors')
    models = []
    for f in os.listdir(MODELS_DIR):
        if f.lower().endswith(valid_extensions):
            models.append(f)
    return sorted(models)


def should_upscale(image_path: str, min_dimension: int = 1024) -> bool:
    """Check if an image's smallest dimension is below the threshold."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return min(width, height) < min_dimension
    except Exception:
        return False


class SpandrelUpscaler:
    def __init__(self, model_name: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = None
        self.scale = 4  # Will be updated when model is loaded

    def load_model(self, model_name: str = None):
        """Load the upscaler model. Reloads if model_name changes."""
        if model_name:
            self.model_name = model_name

        if not self.model_name:
            raise ValueError("No model name specified")

        model_path = os.path.join(MODELS_DIR, self.model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Reload if different model requested
        if self.model is not None:
            # Check if we need to reload
            current_path = getattr(self, '_loaded_path', None)
            if current_path == model_path:
                return  # Already loaded
            self.unload_model()

        print(f"Loading upscaler model: {self.model_name} on {self.device}...")
        try:
            self.model = spandrel.ModelLoader().load_from_file(model_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.scale = self.model.scale
            self._loaded_path = model_path
            print(f"Upscaler loaded. Scale: {self.scale}x")
        except Exception as e:
            print(f"Error loading upscaler model: {e}")
            raise

    def upscale(self, image_path: str, tile_size: int = 512, tile_overlap: int = 32) -> Image.Image:
        """Upscale an image and return the result as a PIL Image.

        Uses tiled processing to avoid VRAM overflow on large images.

        Args:
            image_path: Path to the image file
            tile_size: Size of tiles for processing (default 512)
            tile_overlap: Overlap between tiles for seamless blending (default 32)
        """
        self.load_model()

        try:
            # Load and prepare image
            image = Image.open(image_path).convert("RGB")
            image = ImageOps.exif_transpose(image)

            width, height = image.size

            # For small images, process directly
            if width <= tile_size and height <= tile_size:
                return self._upscale_tensor(image)

            # For larger images, use tiled processing
            return self._upscale_tiled(image, tile_size, tile_overlap)

        except Exception as e:
            print(f"Upscaling error for {image_path}: {e}")
            raise

    def _upscale_tensor(self, image: Image.Image) -> Image.Image:
        """Upscale a PIL image directly (for small images)."""
        import numpy as np

        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)

        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(output)

    def _upscale_tiled(self, image: Image.Image, tile_size: int, overlap: int) -> Image.Image:
        """Upscale a large image using tiled processing to save VRAM."""
        import numpy as np

        width, height = image.size
        scale = self.scale

        # Calculate output dimensions
        out_width = width * scale
        out_height = height * scale

        # Create output array
        output = np.zeros((out_height, out_width, 3), dtype=np.float32)
        weight = np.zeros((out_height, out_width, 1), dtype=np.float32)

        # Calculate tile positions
        step = tile_size - overlap

        for y in range(0, height, step):
            for x in range(0, width, step):
                # Calculate tile boundaries (with clamping)
                x1 = x
                y1 = y
                x2 = min(x + tile_size, width)
                y2 = min(y + tile_size, height)

                # Extract tile
                tile = image.crop((x1, y1, x2, y2))

                # Upscale tile
                tile_array = np.array(tile).astype(np.float32) / 255.0
                tile_tensor = torch.from_numpy(tile_array).permute(2, 0, 1).unsqueeze(0)
                tile_tensor = tile_tensor.to(self.device)

                with torch.no_grad():
                    upscaled_tile = self.model(tile_tensor)

                upscaled_tile = upscaled_tile.squeeze(0).permute(1, 2, 0).cpu().numpy()

                # Calculate output positions
                out_x1 = x1 * scale
                out_y1 = y1 * scale
                out_x2 = out_x1 + upscaled_tile.shape[1]
                out_y2 = out_y1 + upscaled_tile.shape[0]

                # Create blending weights (feather edges for overlap regions)
                tile_h, tile_w = upscaled_tile.shape[:2]
                tile_weight = np.ones((tile_h, tile_w, 1), dtype=np.float32)

                # Feather edges if not at image boundary
                feather = overlap * scale
                if x1 > 0:  # Left edge
                    for i in range(min(feather, tile_w)):
                        tile_weight[:, i, 0] *= i / feather
                if y1 > 0:  # Top edge
                    for i in range(min(feather, tile_h)):
                        tile_weight[i, :, 0] *= i / feather
                if x2 < width:  # Right edge
                    for i in range(min(feather, tile_w)):
                        tile_weight[:, -(i+1), 0] *= i / feather
                if y2 < height:  # Bottom edge
                    for i in range(min(feather, tile_h)):
                        tile_weight[-(i+1), :, 0] *= i / feather

                # Accumulate weighted tile
                output[out_y1:out_y2, out_x1:out_x2] += upscaled_tile * tile_weight
                weight[out_y1:out_y2, out_x1:out_x2] += tile_weight

                # Clear CUDA cache after each tile to prevent memory buildup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Normalize by weights
        output = output / np.maximum(weight, 1e-8)
        output = np.clip(output * 255, 0, 255).astype(np.uint8)

        return Image.fromarray(output)

    def unload_model(self):
        """Unload the model and free VRAM."""
        if self.model is not None:
            del self.model
            self.model = None
            self._loaded_path = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Upscaler model unloaded.")


# Global instance pattern
_global_upscaler = None


def get_upscaler(model_name: str = None) -> SpandrelUpscaler:
    """Get or create the global upscaler instance."""
    global _global_upscaler
    if _global_upscaler is None:
        _global_upscaler = SpandrelUpscaler(model_name)
    elif model_name and model_name != _global_upscaler.model_name:
        # Different model requested, update the name (will reload on next use)
        _global_upscaler.model_name = model_name
    return _global_upscaler
