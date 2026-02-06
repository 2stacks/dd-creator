import torch
import numpy as np
from PIL import Image, ImageOps
import gc
import os

# MobileSAM checkpoint URL
_CHECKPOINT_URL = "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt"
_CHECKPOINT_NAME = "mobile_sam.pt"
_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".models")


class MobileSAMSegmenter:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.predictor = None
        self._current_image_path = None  # For embedding caching

    def _ensure_checkpoint(self):
        """Download MobileSAM checkpoint if not present."""
        os.makedirs(_MODELS_DIR, exist_ok=True)
        checkpoint_path = os.path.join(_MODELS_DIR, _CHECKPOINT_NAME)
        if not os.path.exists(checkpoint_path):
            print(f"Downloading MobileSAM checkpoint to {checkpoint_path}...")
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="dhkim2810/MobileSAM",
                filename="mobile_sam.pt",
                local_dir=_MODELS_DIR,
                local_dir_use_symlinks=False,
            )
            print("MobileSAM checkpoint downloaded.")
        return checkpoint_path

    def load_model(self):
        if self.model is not None:
            return

        checkpoint_path = self._ensure_checkpoint()
        print(f"Loading MobileSAM on {self.device}...")

        from mobile_sam import sam_model_registry, SamPredictor

        self.model = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        self.predictor = SamPredictor(self.model)
        self._current_image_path = None
        print("MobileSAM loaded.")

    def _set_image(self, image_path):
        """Set image for prediction, reusing cached embedding if same image."""
        if self._current_image_path == image_path:
            return

        img = Image.open(image_path).convert("RGB")
        img = ImageOps.exif_transpose(img)
        img_array = np.array(img)
        self.predictor.set_image(img_array)
        self._current_image_path = image_path

    def segment_point(self, image_path, x, y):
        """Segment object at a single point. Returns binary mask (mode 'L')."""
        self.load_model()
        self._set_image(image_path)

        input_point = np.array([[x, y]])
        input_label = np.array([1])  # 1 = foreground

        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # Pick the mask with highest score
        best_idx = np.argmax(scores)
        mask = masks[best_idx]

        binary_mask = (mask > 0).astype(np.uint8) * 255
        return Image.fromarray(binary_mask, mode="L")

    def segment_multi_point(self, image_path, points, labels):
        """Segment using multiple foreground/background points.

        Args:
            image_path: Path to image
            points: List of (x, y) tuples
            labels: List of ints (1=foreground, 0=background)

        Returns:
            Binary mask (mode 'L')
        """
        self.load_model()
        self._set_image(image_path)

        input_points = np.array(points)
        input_labels = np.array(labels)

        masks, scores, _ = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )

        best_idx = np.argmax(scores)
        mask = masks[best_idx]

        binary_mask = (mask > 0).astype(np.uint8) * 255
        return Image.fromarray(binary_mask, mode="L")

    def unload_model(self):
        if self.model is not None:
            del self.model
            del self.predictor
            self.model = None
            self.predictor = None
            self._current_image_path = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("MobileSAM unloaded.")


# Global instance pattern
_global_sam_segmenter = None


def get_sam_segmenter():
    global _global_sam_segmenter
    if _global_sam_segmenter is None:
        _global_sam_segmenter = MobileSAMSegmenter()
    return _global_sam_segmenter
