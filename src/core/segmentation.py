import torch
import numpy as np
from PIL import Image, ImageOps
from transformers import Sam2Model, Sam2Processor
import gc

class SAMSegmenter:
    def __init__(self, model_id="facebook/sam2.1-hiera-large"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.model = None
        self.processor = None
        # Cached state for current image
        self.cached_embeddings = None
        self.cached_image_path = None
        self.original_size = None

    def load_model(self):
        if self.model is None:
            print(f"Loading SAM model: {self.model_id} on {self.device}...")
            try:
                self.processor = Sam2Processor.from_pretrained(self.model_id)
                # Sam2Model.from_pretrained works for sam2.1 models in transformers 4.48+
                self.model = Sam2Model.from_pretrained(self.model_id).to(self.device)
                if self.device == "cuda":
                    # Use float16 for memory efficiency
                    self.model = self.model.half()
                self.model.eval()
                print("SAM model loaded.")
            except Exception as e:
                print(f"Error loading SAM model {self.model_id}: {e}")
                raise e

    def set_image(self, image_path):
        if image_path == self.cached_image_path and self.cached_embeddings is not None:
            return

        self.load_model()
        
        try:
            image = Image.open(image_path).convert("RGB")
            image = ImageOps.exif_transpose(image)
            self.original_size = (image.height, image.width) # (height, width) for processor
            self.cached_image_path = image_path
            
            # Prepare image for the model
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            if self.device == "cuda":
                inputs["pixel_values"] = inputs["pixel_values"].half()
                
            with torch.no_grad():
                # Sam2Model.get_image_embeddings returns a list of tensors
                image_embeddings = self.model.get_image_embeddings(inputs["pixel_values"])
            
            self.cached_embeddings = image_embeddings
            print(f"Computed embeddings for {image_path}")
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            self.cached_embeddings = None
            self.cached_image_path = None

    def segment_from_points(self, points, labels):
        if self.cached_embeddings is None or not points:
            return None

        self.load_model()
        
        try:
            # points: list of [x, y] coordinates
            # labels: list of 1 (foreground) or 0 (background)
            
            # Sam2Model.forward expects:
            # input_points: (batch_size, num_masks, num_points, 2)
            # input_labels: (batch_size, num_masks, num_points)
            
            # We are generating 1 mask per image based on the points
            input_points = torch.tensor([[points]], device=self.device, dtype=torch.float32)
            input_labels = torch.tensor([[labels]], device=self.device, dtype=torch.long)
            
            if self.device == "cuda":
                input_points = input_points.half()

            with torch.no_grad():
                outputs = self.model(
                    image_embeddings=self.cached_embeddings,
                    input_points=input_points,
                    input_labels=input_labels,
                    multimask_output=False
                )
            
            # Post-process masks
            # masks from forward: (batch_size, num_masks, height, width) 
            # for Sam2Model it might be (batch_size, num_masks, h, w) where h,w are 256
            
            masks = self.processor.post_process_masks(
                outputs.pred_masks.cpu(), 
                original_sizes=[self.original_size]
            )
            
            # masks is a list of tensors (one per image in batch)
            # We want the first image's masks -> masks[0]
            # Then the first mask of that image -> masks[0][0]
            # Ensure it is 2D: (H, W)
            raw_mask = masks[0][0]
            if raw_mask.ndim == 3:
                raw_mask = raw_mask.squeeze(0) # Handle (1, H, W)
                
            binary_mask = raw_mask.numpy()
            
            # Ensure bool -> uint8 scaling
            if binary_mask.dtype == bool:
                binary_mask = (binary_mask * 255).astype(np.uint8)
            else:
                 # If it's logits (float), we need to threshold, but post_process_masks usually does that or returns bool
                 binary_mask = (binary_mask > 0).astype(np.uint8) * 255
            
            # Convert to PIL
            mask_image = Image.fromarray(binary_mask)
            return mask_image

        except Exception as e:
            print(f"Segmentation error: {e}")
            return None

    def unload_model(self):
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self.cached_embeddings = None
            self.cached_image_path = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("SAM model unloaded.")

# Global instance pattern
_global_segmenter = None

def get_segmenter(model_id="facebook/sam2.1-hiera-large"):
    global _global_segmenter
    if _global_segmenter is None:
        _global_segmenter = SAMSegmenter(model_id)
    elif _global_segmenter.model_id != model_id:
        _global_segmenter.unload_model()
        _global_segmenter = SAMSegmenter(model_id)
    return _global_segmenter